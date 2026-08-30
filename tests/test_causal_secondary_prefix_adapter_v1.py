from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3, ControlBandContract
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.losses.broadband_loss import CausalFIRPathData
from deep_anc.models.hybrid_anc import HybridANCNet
from deep_anc.train.causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
    CausalSecondaryPrefixAdapterV1,
)
from deep_anc.train.full_octave_causal_plant_binding_v4 import (
    FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4,
    FullOctaveCausalPlantBindingV4,
)
from deep_anc.train.full_octave_v3_admission import (
    V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _operator(
    *,
    role: str,
    fir: tuple[float, ...],
    coarse_delay: int,
    handoff: int,
    authority_sha: str,
) -> CausalFIRPathData:
    values = np.ascontiguousarray(np.asarray(fir, dtype=np.float64))
    return CausalFIRPathData(
        role=role,  # type: ignore[arg-type]
        post_onset_fir=values,
        coarse_delay_samples=coarse_delay,
        fractional_delay_samples=0.0,
        support_samples=len(values),
        sample_rate=48_000,
        handoff_extra_samples=handoff,
        operator_file_sha256=_sha(f"{role}-file"),
        operator_internal_sha256=_sha(f"{role}-internal"),
        fir_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        authority_sha256=authority_sha,
        source_path=f"future/full_octave/{role}.npz",
    )


def _binding_kwargs(**overrides: object) -> dict[str, object]:
    authority_sha = _sha("plant-authority")
    primary_fir = (0.0, -0.25, 1.0, 0.1)
    delays = PlantDelays(
        primary_delay_samples=10,
        secondary_delay_samples=6,
        handoff_samples=256,
        sample_rate=48_000,
    )
    timing = TrainingTimingContract.derive(primary_fir=primary_fir, plant_delays=delays)
    contract = BroadbandFullOctaveContractV3.canonical()
    values: dict[str, object] = {
        "control_band_contract": contract,
        "control_band_contract_sha256": contract.digest(),
        "training_timing_contract": timing,
        "training_timing_contract_sha256": timing.digest(),
        "primary_operator": _operator(
            role="primary",
            fir=primary_fir,
            coarse_delay=10,
            handoff=0,
            authority_sha=authority_sha,
        ),
        "secondary_operator": _operator(
            role="secondary",
            fir=(0.5, -0.25, 0.125),
            coarse_delay=6,
            handoff=256,
            authority_sha=authority_sha,
        ),
        "verified_physical_subbands_hz": contract.physical_identification_subbands_hz,
        "raw_capture_sha256": _sha("raw"),
        "analysis_sha256": _sha("analysis"),
        "primary_raw_capture_sha256": _sha("raw"),
        "secondary_raw_capture_sha256": _sha("raw"),
        "primary_analysis_sha256": _sha("analysis"),
        "secondary_analysis_sha256": _sha("analysis"),
        "plant_authority_sha256": authority_sha,
        "electrical_witness_receipt_sha256": _sha("witness"),
        "err_channel_index": 0,
        "err_channel_selection_sha256": _sha("err-channel-selection"),
        "reference_channel_index": 0,
        "reference_channel_selection_sha256": _sha("reference-channel-selection"),
        "authority_schema": FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4,
        "block_size": 256,
        "schema_version": "full_octave_causal_plant_binding_v4",
    }
    values.update(overrides)
    return values


def _binding(**overrides: object) -> FullOctaveCausalPlantBindingV4:
    return FullOctaveCausalPlantBindingV4._for_test_fixture(
        **_binding_kwargs(**overrides)
    )


@pytest.fixture()
def binding() -> FullOctaveCausalPlantBindingV4:
    return _binding()


class _StreamingGainController(torch.nn.Module):
    """``forward`` 호출을 허용하지 않는 stateful streaming fixture."""

    hop = 128
    in_channels = 2
    context = 256

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.step_calls = 0

    def forward(self, _value: torch.Tensor) -> torch.Tensor:  # pragma: no cover - must not run
        raise AssertionError("adapter는 controller.forward()를 호출하면 안 됩니다")

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros((batch, 1, 1), device=device)

    def streaming_step(
        self, x_block: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.step_calls += 1
        y = self.gain * (x_block[:, :1] - 0.35 * x_block[:, 1:2]) + 0.1 * state
        return y, y[..., -1:]


class _BFloatStreamingGainController(_StreamingGainController):
    def streaming_step(
        self, x_block: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, next_state = super().streaming_step(x_block, state)
        return y.to(torch.bfloat16), next_state.to(torch.bfloat16)


def _adapter(binding: FullOctaveCausalPlantBindingV4) -> CausalSecondaryPrefixAdapterV1:
    return CausalSecondaryPrefixAdapterV1._for_test_fixture(binding)


def _batch(
    adapter: CausalSecondaryPrefixAdapterV1,
    *,
    prefix_samples: int = 512,
    target_samples: int = 512,
    x_target: torch.Tensor | None = None,
) -> CausalPrefixBatchV1:
    torch.manual_seed(8)
    lead = int(adapter.binding.training_timing_contract.digital_reference_lead_samples)
    clean_timeline = torch.randn(
        (1, 1, prefix_samples + target_samples + lead), dtype=torch.float32
    ) * 0.03
    clean_preview = clean_timeline[..., lead : lead + prefix_samples + target_samples]
    # controller ch0에는 input-only mic noise가 들어가도 P(n)는 바뀌지 않아야 한다.
    x_prefix = torch.cat(
        (
            clean_preview[..., :prefix_samples] + 0.002 * torch.randn((1, 1, prefix_samples)),
            0.01 * torch.randn((1, 1, prefix_samples)),
        ),
        dim=1,
    )
    if x_target is None:
        x_target = torch.cat(
            (
                clean_preview[..., prefix_samples:]
                + 0.002 * torch.randn((1, 1, target_samples)),
                0.01 * torch.randn((1, 1, target_samples)),
            ),
            dim=1,
        )
    source = (_sha("unseen-source"),)
    return CausalPrefixBatchV1(
        x_prefix=x_prefix,
        x_target=x_target,
        source_sha256=source,
        clean_playback_source_sha256=source,
        clean_playback_timeline=clean_timeline,
        controller_reference_preaugmentation=clean_preview,
        training_timing_contract_sha256=adapter.binding.training_timing_contract_sha256,
        segment_prefix_start_samples=(0,),
        segment_target_start_samples=(prefix_samples,),
        global_sample_indices=(17,),
        state_origin=CausalPrefixStateOriginV1(
            kind="segment_start_zero_state",
            binding_sha256=adapter.binding_sha256,
            source_sha256=source,
        ),
    )


def _manual_stream(
    controller: _StreamingGainController, value: torch.Tensor
) -> torch.Tensor:
    state = controller.init_states(batch=value.shape[0], device=value.device)
    outputs: list[torch.Tensor] = []
    for start in range(0, value.shape[-1], 256):
        output, state = controller.streaming_step(value[..., start : start + 256], state)
        outputs.append(output.float())
    return torch.cat(outputs, dim=-1)


def _direct_causal(value: torch.Tensor, path: CausalFIRPathData) -> torch.Tensor:
    """FFT implementation을 재사용하지 않는 direct-time-domain causal oracle."""

    assert value.ndim == 3 and value.shape[1] == 1
    samples = int(value.shape[-1])
    delayed = torch.zeros_like(value, dtype=torch.float32)
    delay = int(path.base_delay_samples)
    if delay < samples:
        delayed[..., delay:] = value.float()[..., : samples - delay]
    output = torch.zeros_like(delayed)
    for tap, coefficient in enumerate(path.post_onset_fir.tolist()):
        if tap < samples:
            output[..., tap:] = output[..., tap:] + float(coefficient) * delayed[
                ..., : samples - tap
            ]
    return output


def _common_physical_transform(value: torch.Tensor) -> torch.Tensor:
    """공통 gain/polarity/EQ를 clean playback ``n``에 적용한 작은 oracle.

    mic-only noise/hum/dropout은 이 transform 뒤 controller input에만 더해야 한다.
    이 helper는 common causal EQ ``[0.8, -0.2]``와 polarity/gain ``-0.6``을 같은
    physical playback timeline에 적용한다.
    """

    transformed = 0.8 * value
    transformed[..., 1:] = transformed[..., 1:] - 0.2 * value[..., :-1]
    return -0.6 * transformed


def test_public_constructor_and_adapter_reject_test_fixture(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    with pytest.raises(TypeError, match="raw-bound publisher"):
        FullOctaveCausalPlantBindingV4(**_binding_kwargs())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="test fixture"):
        CausalSecondaryPrefixAdapterV1(binding)
    assert _adapter(binding).binding.fixture_only is True


def test_binding_clones_readonly_fir_bytes_and_external_mutation_cannot_change_it() -> None:
    values = _binding_kwargs()
    external_secondary = values["secondary_operator"]
    assert isinstance(external_secondary, CausalFIRPathData)
    binding = FullOctaveCausalPlantBindingV4._for_test_fixture(**values)
    before = binding.secondary_operator.post_onset_fir.copy()
    external_secondary.post_onset_fir[0] = 9.0

    assert np.array_equal(binding.secondary_operator.post_onset_fir, before)
    assert binding.secondary_operator.post_onset_fir.flags.writeable is False
    with pytest.raises(ValueError):
        binding.secondary_operator.post_onset_fir[0] = 9.0


def test_full_prefix_primary_and_secondary_crop_match_independent_direct_oracle(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    batch = _batch(adapter)
    controller = _StreamingGainController()
    result = adapter(controller, batch)

    reference_controller = _StreamingGainController()
    reference_controller.load_state_dict(controller.state_dict())
    x_full = torch.cat((batch.x_prefix, batch.x_target), dim=-1)
    y_full = _manual_stream(reference_controller, x_full)
    expected_primary = _direct_causal(
        batch.clean_playback_timeline[..., :1024], binding.primary_operator
    )[..., 512:]
    expected_secondary = _direct_causal(y_full, binding.secondary_operator)[..., 512:]
    lead = int(binding.training_timing_contract.digital_reference_lead_samples)

    assert torch.allclose(result.y_prefix, y_full[..., :512])
    assert torch.allclose(result.y_target, y_full[..., 512:])
    assert torch.equal(
        result.clean_reference_preview,
        batch.clean_playback_timeline[..., lead : lead + 1024],
    )
    assert torch.allclose(result.primary_target, expected_primary, atol=2e-7, rtol=2e-6)
    assert torch.allclose(result.secondary_target, expected_secondary, atol=2e-7, rtol=2e-6)
    # 서로 다른 causal FIR oracle 경로의 FP32 누적 순서가 Torch 버전별로 수십
    # 나노 단위 달라질 수 있다. adapter 내부의 composition은 먼저 exact하게
    # 검증하고, 독립 direct oracle과의 비교에는 위 crop 검증과 같은 허용오차를 쓴다.
    assert torch.equal(result.error_target, result.primary_target + result.secondary_target)
    assert torch.allclose(
        result.error_target,
        expected_primary + expected_secondary,
        atol=2e-7,
        rtol=2e-6,
    )
    assert result.primary_target.dtype == torch.float32
    assert result.secondary_target.dtype == torch.float32
    assert result.binding_sha256 == binding.digest()
    assert controller.step_calls == 4
    assert binding.canonical_training_eligible is False


def test_nonzero_digital_lead_primary_uses_clean_playback_not_preview(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    """``x_ref=n(t+K)``이어도 P에는 ``n(t)``만 들어가야 한다."""

    adapter = _adapter(binding)
    template = _batch(adapter)
    prefix_samples = int(template.x_prefix.shape[-1])
    target_samples = int(template.x_target.shape[-1])
    total_samples = prefix_samples + target_samples
    lead = int(binding.training_timing_contract.digital_reference_lead_samples)
    assert lead > 0

    # Non-periodic impulses ensure that P(n) and an incorrect P(n(t+K)) cannot
    # accidentally coincide after the target crop.
    clean_timeline = torch.zeros_like(template.clean_playback_timeline)
    clean_timeline[..., 73] = 0.19
    clean_timeline[..., prefix_samples + 300] = -0.61
    preview = clean_timeline[..., lead : lead + total_samples]
    controller_input = torch.cat((preview, torch.zeros_like(preview)), dim=1)
    batch = replace(
        template,
        x_prefix=controller_input[..., :prefix_samples],
        x_target=controller_input[..., prefix_samples:],
        clean_playback_timeline=clean_timeline,
        controller_reference_preaugmentation=preview,
    )

    result = adapter(_StreamingGainController(), batch)
    correct_primary = _direct_causal(
        clean_timeline[..., :total_samples], binding.primary_operator
    )[..., prefix_samples:]
    wrong_preview_primary = _direct_causal(preview, binding.primary_operator)[
        ..., prefix_samples:
    ]

    assert torch.equal(result.clean_reference_preview, preview)
    assert torch.allclose(result.primary_target, correct_primary, atol=2e-7, rtol=2e-6)
    assert not torch.allclose(correct_primary, wrong_preview_primary)


def test_common_gain_polarity_eq_precedes_input_only_augmentation(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    """공통 physical transform은 clean P timeline과 pre-augmentation preview에 함께 남는다."""

    adapter = _adapter(binding)
    template = _batch(adapter)
    prefix_samples = int(template.x_prefix.shape[-1])
    target_samples = int(template.x_target.shape[-1])
    total_samples = prefix_samples + target_samples
    lead = int(binding.training_timing_contract.digital_reference_lead_samples)
    transformed_clean = _common_physical_transform(template.clean_playback_timeline)
    transformed_preview = transformed_clean[..., lead : lead + total_samples]
    # Input-only augmentation은 transformed preview 뒤에만 더한다.
    controller_input = torch.cat(
        (
            transformed_preview + 0.003,
            torch.full_like(transformed_preview, -0.002),
        ),
        dim=1,
    )
    batch = replace(
        template,
        x_prefix=controller_input[..., :prefix_samples],
        x_target=controller_input[..., prefix_samples:],
        clean_playback_timeline=transformed_clean,
        controller_reference_preaugmentation=transformed_preview,
    )

    result = adapter(_StreamingGainController(), batch)
    expected_primary = _direct_causal(
        transformed_clean[..., :total_samples], binding.primary_operator
    )[..., prefix_samples:]

    assert torch.equal(result.clean_reference_preview, transformed_preview)
    assert torch.allclose(result.primary_target, expected_primary, atol=2e-7, rtol=2e-6)
    assert not torch.equal(transformed_preview, template.controller_reference_preaugmentation)


def test_target_only_secondary_filter_is_not_an_equivalent_crop(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    result = adapter(_StreamingGainController(), _batch(adapter))
    target_only = _direct_causal(result.y_target, binding.secondary_operator)
    assert not torch.allclose(result.secondary_target, target_only)
    assert torch.max(torch.abs(result.secondary_target - target_only)) > 1.0e-6


def test_input_only_reference_noise_or_dropout_cannot_change_primary_target(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    clean = _batch(adapter)
    altered_prefix = clean.x_prefix.clone()
    altered_target = clean.x_target.clone()
    altered_prefix[:, 0] = 0.0  # ref dropout
    altered_target[:, 0] = altered_target[:, 0] + 1.0  # severe input-only mic contamination
    altered = replace(clean, x_prefix=altered_prefix, x_target=altered_target)

    baseline = adapter(_StreamingGainController(), clean)
    noisy = adapter(_StreamingGainController(), altered)
    assert torch.equal(baseline.primary_target, noisy.primary_target)
    assert torch.equal(baseline.clean_reference_preview, noisy.clean_reference_preview)
    assert not torch.equal(baseline.y_target, noisy.y_target)


def test_future_target_does_not_change_prefix_output_and_forward_is_never_called(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    original = _batch(adapter)
    mutated_target = original.x_target.clone()
    mutated_target[..., -1] += 1.0
    changed = _batch(adapter, x_target=mutated_target)

    first = adapter(_StreamingGainController(), original)
    second = adapter(_StreamingGainController(), changed)
    assert torch.equal(first.y_prefix, second.y_prefix)
    assert not torch.equal(first.y_target, second.y_target)


def test_target_loss_gradient_reaches_prefix_and_controller_parameter(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    batch = _batch(adapter)
    prefix = batch.x_prefix.detach().clone().requires_grad_(True)
    batch = replace(batch, x_prefix=prefix)
    controller = _StreamingGainController()
    result = adapter(controller, batch)
    loss = result.error_target.square().mean()
    loss.backward()

    assert controller.gain.grad is not None
    assert torch.isfinite(controller.gain.grad)
    assert controller.gain.grad.abs() > 0
    assert prefix.grad is not None
    assert torch.isfinite(prefix.grad).all()
    assert torch.count_nonzero(prefix.grad) > 0


def test_bfloat_controller_output_is_cast_to_fp32_before_both_plants(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    result = adapter(_BFloatStreamingGainController(), _batch(adapter))
    assert result.y_target.dtype == torch.float32
    assert result.primary_target.dtype == torch.float32
    assert result.secondary_target.dtype == torch.float32


def test_prefix_block_timing_and_context_contract_fail_closed(
    binding: FullOctaveCausalPlantBindingV4,
) -> None:
    adapter = _adapter(binding)
    with pytest.raises(ValueError, match="history"):
        adapter(_StreamingGainController(), _batch(adapter, prefix_samples=256))

    batch = _batch(adapter)
    nonmultiple = replace(batch, x_target=batch.x_target[..., :-1])
    with pytest.raises(ValueError, match="256-sample"):
        adapter(_StreamingGainController(), nonmultiple)

    bad_origin = replace(
        batch,
        segment_prefix_start_samples=(1,),
        segment_target_start_samples=(513,),
    )
    with pytest.raises(ValueError, match="segment sample 0"):
        adapter(_StreamingGainController(), bad_origin)

    with pytest.raises(ValueError, match="timing-v2"):
        adapter(
            _StreamingGainController(),
            replace(batch, training_timing_contract_sha256=_sha("wrong-timing")),
        )

    nonfinite = replace(batch, x_prefix=torch.full_like(batch.x_prefix, float("nan")))
    with pytest.raises(ValueError, match="NaN/Inf"):
        adapter(_StreamingGainController(), nonfinite)

    with pytest.raises(ValueError, match=r"prefix \+ target \+ derived lead"):
        adapter(
            _StreamingGainController(),
            replace(batch, clean_playback_timeline=batch.clean_playback_timeline[..., :-1]),
        )

    shifted_preview = torch.cat(
        (
            batch.controller_reference_preaugmentation[..., 1:],
            torch.zeros_like(batch.controller_reference_preaugmentation[..., :1]),
        ),
        dim=-1,
    )
    with pytest.raises(ValueError, match="derived lead preview"):
        adapter(
            _StreamingGainController(),
            replace(batch, controller_reference_preaugmentation=shifted_preview),
        )

    too_wide = _StreamingGainController()
    too_wide.context = 257
    with pytest.raises(ValueError, match="context"):
        adapter(too_wide, batch)


def test_binding_rejects_legacy_static_or_timing_mismatch() -> None:
    with pytest.raises(ValueError, match="legacy/static"):
        _binding(authority_schema="external_electrical_witness_admission_v1")

    with pytest.raises(ValueError, match="canonical v3"):
        _binding(control_band_contract=ControlBandContract.stage1_strict())

    with pytest.raises(ValueError, match="canonical 88.388"):
        _binding(verified_physical_subbands_hz=((150.0, 1600.0),))

    base = _binding_kwargs()
    authority = str(base["plant_authority_sha256"])
    broken_secondary = _operator(
        role="secondary",
        fir=(0.5, -0.25, 0.125),
        coarse_delay=7,
        handoff=256,
        authority_sha=authority,
    )
    with pytest.raises(ValueError, match="secondary delay"):
        _binding(secondary_operator=broken_secondary)

    with pytest.raises(ValueError, match="같은 immutable raw"):
        _binding(primary_raw_capture_sha256=_sha("other-raw"))

    with pytest.raises(ValueError, match="payload/SHA"):
        _binding(training_timing_contract_sha256=_sha("wrong-timing"))

    wrong_peak_primary = _operator(
        role="primary",
        fir=(1.0, -0.25, 0.1, 0.0),
        coarse_delay=10,
        handoff=0,
        authority_sha=authority,
    )
    with pytest.raises(ValueError, match="primary FIR peak"):
        _binding(primary_operator=wrong_peak_primary)

    with pytest.raises(ValueError, match="ERR channel index"):
        _binding(err_channel_index="0")


@pytest.mark.parametrize("config_name", ("model_tiny.yaml", "model_base.yaml"))
def test_actual_tiny_and_base_streaming_apis_fit_256_block_contract(
    binding: FullOctaveCausalPlantBindingV4,
    config_name: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / config_name).read_text(encoding="utf-8"))
    model = HybridANCNet(config).eval()
    adapter = _adapter(binding)
    with torch.no_grad():
        result = adapter(model, _batch(adapter))
    assert model.hop == 128 and model.context == 256
    assert result.y_target.shape == (1, 1, 512)
    assert result.secondary_target.dtype == torch.float32


def test_module_is_device_agnostic_and_does_not_open_v3_training_gate() -> None:
    import deep_anc.train.causal_secondary_prefix_adapter_v1 as module

    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"sounddevice", "subprocess", "cuda", "realtime", "audio_io"}.isdisjoint(imports)
    assert ".forward(" not in source
    # Adapter만 있던 초기 상태와 달리, 같은 binding의 v3 loss/FxLMS consumer가
    # 별도 regression으로 연결됐다. physical/canonical admission은 여전히 이
    # fixture가 아니라 raw-bound authority에서 따로 닫힌다.
    assert V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED is True
