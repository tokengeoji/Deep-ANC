"""실제 잔류 ERR로 FxNLMS가 보정하는지와 잘못된 artifact 혼용을 검증한다."""

import numpy as np
import pytest

from deep_anc.config import load_runtime_config
from deep_anc.realtime import engines
from deep_anc.realtime.engines import (
    FxLMSEngine, HybridEngine, checkpoint_reference_mode,
    checkpoint_digital_reference_lead_samples, validate_secondary_calibration,
)


class FixedNeural:
    """S=-0.5에서 d=0.02*x의 일부만 제거하는 고정 제어기."""

    reference_mode = "acoustic"
    digital_reference_lead_samples = 0

    def __init__(self, hop=32, **_):
        self.hop = hop
        self.calls = 0

    def reset(self):
        self.calls = 0

    def step(self, ref, err):
        self.calls += 1
        return np.float32(0.02) * ref


def adaptive_engine(tmp_path, hop=32):
    path = tmp_path / "secondary.npz"
    np.savez(path, fir=np.array([-0.5], dtype=np.float32), delay_samples=0, sample_rate=48000)
    return FxLMSEngine(
        str(path), {"control_length": 1, "mu": 0.1, "leakage": 0.0},
        hop=hop, handoff_extra_samples=hop,
    )


def test_hybrid_corrects_residual_with_actual_delayed_error_and_polarity(tmp_path):
    hop = 32
    hybrid = HybridEngine(FixedNeural(hop), adaptive_engine(tmp_path, hop))
    rng = np.random.default_rng(4)
    previous_ref = np.zeros(hop, dtype=np.float32)
    previous_y = np.zeros(hop, dtype=np.float32)
    errors = []
    # 합성 플랜트: 1 hop 제어 지연만큼 REF가 먼저 관측되는 조건.
    for i in range(240):
        ref = rng.normal(0.0, 0.1, hop).astype(np.float32)
        error = 0.02 * previous_ref - 0.5 * previous_y
        hybrid.set_adapt_enabled(i >= 2)
        y = hybrid.step(ref, error)
        errors.append(float(np.mean(error**2)))
        previous_ref, previous_y = ref, y
    assert np.mean(errors[-20:]) < np.mean(errors[2:12]) * 1e-3
    assert hybrid.adaptive.controller.w[0] == pytest.approx(0.02, abs=1e-5)
    assert hybrid.last_adaptation.adapted
    hybrid.reset()
    assert hybrid.neural.calls == 0
    assert not hybrid.adaptive.adapt
    np.testing.assert_array_equal(hybrid.adaptive.controller.w, 0)


def test_hybrid_starts_frozen_and_does_not_hide_combined_output_clipping(tmp_path):
    hybrid = HybridEngine(FixedNeural(), adaptive_engine(tmp_path))
    ref = np.full(32, 20.0, dtype=np.float32)
    error = np.ones(32, dtype=np.float32)
    for _ in range(3):
        output = hybrid.step(ref, error)
    assert np.max(output) > 0.2  # 합산 뒤 런타임 리미터가 clipping을 관측해야 한다.
    assert hybrid.adaptive.controller.update_count == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_bad_neural_output_resets_both_paths_and_disables_adaptation(tmp_path, bad):
    neural = FixedNeural()
    hybrid = HybridEngine(neural, adaptive_engine(tmp_path))
    hybrid.set_adapt_enabled(True)
    hybrid.adaptive.controller.w[:] = 1.0
    neural.step = lambda ref, err: np.full(32, bad, dtype=np.float32)
    with pytest.raises(ValueError, match="신경망 출력"):
        hybrid.step(np.ones(32, dtype=np.float32), np.zeros(32, dtype=np.float32))
    assert not hybrid.adaptive.adapt
    np.testing.assert_array_equal(hybrid.adaptive.controller.w, 0)


def test_hybrid_rejects_mismatched_hops_and_input_shapes(tmp_path):
    with pytest.raises(ValueError, match="hop"):
        HybridEngine(FixedNeural(16), adaptive_engine(tmp_path))
    hybrid = HybridEngine(FixedNeural(), adaptive_engine(tmp_path))
    with pytest.raises(ValueError, match="모양"):
        hybrid.step(np.zeros((1, 32)), np.zeros(32))


def test_reference_metadata_is_not_inferred_from_zero_lead():
    assert checkpoint_reference_mode({"cfg": {"digital_reference_lead_samples": 0}}) is None
    assert checkpoint_reference_mode({"cfg": {"data": {"reference_mode": "acoustic"}}}) == "acoustic"
    assert checkpoint_reference_mode({"cfg": {"reference_mode": "digital"}}) == "digital"
    with pytest.raises(ValueError, match="reference_mode"):
        checkpoint_reference_mode({"cfg": {"reference_mode": "mic"}})


def test_conflicting_nested_checkpoint_metadata_is_rejected():
    with pytest.raises(ValueError, match="상위/중첩 reference_mode"):
        checkpoint_reference_mode({"cfg": {
            "reference_mode": "acoustic", "data": {"reference_mode": "digital"},
        }})
    with pytest.raises(ValueError, match="상위/중첩 digital_reference_lead_samples"):
        checkpoint_digital_reference_lead_samples({"cfg": {
            "digital_reference_lead_samples": 0,
            "data": {"digital_reference_lead_samples": 109},
        }})


@pytest.mark.parametrize("controller", ["dl", "hybrid"])
@pytest.mark.parametrize("mode", [None, "digital"])
def test_mic_rejects_non_acoustic_neural_artifact(monkeypatch, controller, mode):
    cfg = load_runtime_config("configs/runtime_acoustic_hybrid.yaml")
    cfg["controller"] = controller
    model = FixedNeural(256)
    model.reference_mode = mode
    monkeypatch.setattr(engines, "OrtEngine", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="acoustic으로 학습"):
        engines.build_engine(cfg)


def test_builds_acoustic_baseline_and_explicit_acoustic_hybrid(monkeypatch):
    cfg = load_runtime_config("configs/runtime_acoustic.yaml")
    assert cfg["reference"] == "mic" and not cfg["noise"]["enabled"]
    assert not cfg["start_on"] and cfg["digital_reference_lead_samples"] == 0
    baseline = engines.build_engine(cfg)
    assert isinstance(baseline, FxLMSEngine) and not baseline.adapt
    cfg = load_runtime_config("configs/runtime_acoustic_hybrid.yaml")
    monkeypatch.setattr(engines, "OrtEngine", lambda *args, **kwargs: FixedNeural(kwargs["hop"]))
    hybrid = engines.build_engine(cfg)
    assert isinstance(hybrid, HybridEngine)
    assert hybrid.secondary_delay_samples == baseline.secondary_delay_samples
    assert hybrid.reference_mode == "acoustic"


def test_unknown_controller_is_not_silently_treated_as_dl():
    with pytest.raises(ValueError, match="controller"):
        engines.build_engine({"controller": "hybird"})


def calibration_config(tmp_path, *, reference="mic", **metadata):
    """실제 NPZ 로드→엔진 생성 경로에서 측정 조건 검증을 확인한다."""
    cfg = load_runtime_config("configs/runtime_acoustic.yaml")
    cfg["reference"] = reference
    path = tmp_path / "calibration.npz"
    arrays = {
        "fir": np.array([-0.5], dtype=np.float32), "delay_samples": 12,
        "sample_rate": 48000, "calibration_block_size": 256,
        "calibration_latency": "low",
    }
    for key, value in metadata.items():
        if value is None:
            arrays.pop(key, None)
        else:
            arrays[key] = value
    np.savez(path, **arrays)
    cfg["duct"]["secondary_path"]["npz"] = str(path)
    return cfg


@pytest.mark.parametrize("controller", ["fxlms", "hybrid"])
@pytest.mark.parametrize("metadata,message", [
    ({"calibration_block_size": None}, "block_size"),
    ({"calibration_latency": None}, "latency"),
    ({"calibration_block_size": 512}, "block_size"),
    ({"calibration_latency": "high"}, "latency"),
    ({"sample_rate": 44100}, "sample_rate"),
])
def test_acoustic_calibration_fails_before_neural_loading(tmp_path, monkeypatch, controller, metadata, message):
    cfg = calibration_config(tmp_path, **metadata)
    cfg["controller"] = controller
    cfg["engine"] = {"type": "ort", "onnx": "/must/not/load.onnx"}
    def forbidden(*args, **kwargs):
        pytest.fail("잘못된 S(z)를 거부하기 전에 신경망을 로드했습니다")
    monkeypatch.setattr(engines, "OrtEngine", forbidden)
    with pytest.raises(ValueError, match=message):
        engines.build_engine(cfg)


def test_acoustic_validated_calibration_and_shared_helper(tmp_path):
    from deep_anc.baselines.fxlms_core import load_secondary_path

    cfg = calibration_config(tmp_path)
    model = load_secondary_path(cfg["duct"]["secondary_path"]["npz"])
    assert validate_secondary_calibration(cfg, model) is None
    baseline = engines.build_engine(cfg)
    assert baseline.secondary_delay_samples == 12 + 256


@pytest.mark.parametrize("metadata", [
    {"calibration_block_size": None, "calibration_latency": None},
    {"calibration_block_size": 0, "calibration_latency": "legacy-npy"},
])
def test_digital_legacy_missing_calibration_metadata_remains_compatible(tmp_path, metadata):
    cfg = calibration_config(tmp_path, reference="digital", **metadata)
    assert isinstance(engines.build_engine(cfg), FxLMSEngine)


@pytest.mark.parametrize("metadata,message", [
    ({"calibration_block_size": 512}, "block_size"),
    ({"calibration_latency": "high"}, "latency"),
    ({"sample_rate": 44100}, "sample_rate"),
    ({"calibration_latency": "invalid"}, "latency"),
])
def test_digital_known_calibration_mismatch_is_not_ignored(tmp_path, metadata, message):
    cfg = calibration_config(tmp_path, reference="digital", **metadata)
    with pytest.raises(ValueError, match=message):
        engines.build_engine(cfg)
