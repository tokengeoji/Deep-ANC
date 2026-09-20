"""독립 검토 회귀: 원본 보존·진단 정보 조건·실패 및 출력 경로 보호."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepanc.calibration import DEFAULT_RIR, load_secondary_path
from deep_anc.eval import sfanc_stress as stress
from deep_anc.train.sfanc_experiment import save_selector
from deep_anc.train.sfanc_selector import RefSpectrumSelector


def make_artifact(root, *, changed_secondary=False):
    root.mkdir()
    bank = np.zeros((2, 8))
    bank[1] = [.02, -.01, .005, 0, 0, 0, 0, 0]
    secondary = load_secondary_path(sample_rate=16000).copy()
    if changed_secondary:
        secondary[0] = 1.0
    with (root / "bank.npz").open("xb") as stream:
        np.savez(stream, coefficients=bank, secondary=secondary)
    bank_sha = hashlib.sha256((root / "bank.npz").read_bytes()).hexdigest()
    model = RefSpectrumSelector(2, 33)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    selector_sha = save_selector(model, root, bank_sha,
                                SimpleNamespace(n_fft=64, window_samples=512, sample_rate=16000))
    prior = {"config": {"secondary_kind": "omap_measured", "regularization": 1e-7,
                        "effort_penalty": .002},
             "training": {"fixed_candidate_from_train": 1},
             "artifacts": {"bank.npz": bank_sha, "selector.pt": selector_sha}}
    (root / "report.json").write_text(json.dumps(prior))
    return root


def small_config():
    return stress.StressConfig(samples=2048, fit_samples=2048,
                              primary_delays=(48,), additional_delays=(3,))


def test_independent_run_preserves_source_bytes_and_uses_separate_fit_reference(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "artifact")
    originals = {path: path.read_bytes() for path in root.iterdir()}
    original_rir = DEFAULT_RIR.read_bytes()
    original_threads = torch.get_num_threads()
    x = np.random.default_rng(734).normal(0, .04, 2048)
    fit_inputs = []
    real_fit = stress.fit_control_fir

    def record_fit(reference, *args, **kwargs):
        fit_inputs.append(reference.copy())
        return real_fit(reference, *args, **kwargs)

    monkeypatch.setattr(stress, "fit_control_fir", record_fit)
    monkeypatch.setattr(stress, "diagnostic_sources",
                        lambda *args: iter([("review_signal", x, {"measured_ref_err": False})]))
    report = stress.run_stress(root, tmp_path / "output", small_config())
    assert len(fit_inputs) == 1
    assert not np.array_equal(fit_inputs[0], x)
    assert report["fitted_reference"]["seed_namespace"] == [small_config().seed, 300]
    assert report["model_retrained"] is False and report["test_used_for_tuning"] is False
    assert report["secondary"]["taps"] == 500
    assert report["secondary"]["all_taps_gain_sign_delay_preserved"] is True
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert DEFAULT_RIR.read_bytes() == original_rir
    assert torch.get_num_threads() == original_threads
    # 첫 결정 지연·전환과 P/S/FIR memory가 모두 공통 평가 시작점에 포함된다.
    starts = {row["start_sample"] for row in report["metrics"] if row["period"] == "after_common_startup"}
    expected = 512 + 64 + 128 + max(48+73, 3+500+8-2)
    assert starts == {expected}
    assert "not same information" in report["protocol"]["scenario_fitted_fir"]


def test_rebound_valid_artifact_with_changed_secondary_is_rejected_before_output_creation(tmp_path):
    root = make_artifact(tmp_path / "changed", changed_secondary=True)
    before = {path: path.read_bytes() for path in root.iterdir()}
    out = tmp_path / "not_created"
    with pytest.raises(ValueError, match="원본"):
        stress.run_stress(root, out, small_config())
    assert not out.exists()
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("kind", ["existing", "leaf_symlink", "parent_symlink"])
def test_output_protection_precedes_artifact_loading(tmp_path, monkeypatch, kind):
    real = tmp_path / "real"
    real.mkdir()
    marker = real / "marker"
    marker.write_bytes(b"preserve")
    if kind == "existing":
        out, error = real, FileExistsError
    elif kind == "leaf_symlink":
        out = tmp_path / "linked"
        out.symlink_to(real, target_is_directory=True)
        error = ValueError
    else:
        parent = tmp_path / "linked"
        parent.symlink_to(real, target_is_directory=True)
        out, error = parent / "new", ValueError
    monkeypatch.setattr(stress, "load_selector", lambda _: pytest.fail("출력 경로부터 거부해야 함"))
    with pytest.raises(error):
        stress.run_stress(tmp_path / "missing_artifact", out, small_config())
    assert marker.read_bytes() == b"preserve"
    assert len(list(real.iterdir())) == 1


def test_source_failure_restores_threads_without_success_report_or_original_mutation(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "artifact")
    before = {path: path.read_bytes() for path in root.iterdir()}
    original_threads = torch.get_num_threads()

    def broken_sources(*args):
        raise ValueError("source 검증 실패")

    monkeypatch.setattr(stress, "diagnostic_sources", broken_sources)
    out = tmp_path / "failed_output"
    with pytest.raises(ValueError, match="source"):
        stress.run_stress(root, out, small_config())
    assert out.is_dir() and not (out / "report.json").exists()
    assert torch.get_num_threads() == original_threads
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_selector_may_modify_its_copy_but_cannot_change_reference_or_prior_audio():
    x = np.ones(29)
    original = x.copy()

    def choose(past):
        past[:] = -1000
        return 1

    result = stress.selected_stream(x, [[0.], [.1]], choose, window_samples=8,
                                    selector_delay_samples=2, control_limit=None)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(result["control"][:10], 0)
    np.testing.assert_allclose(result["control"][10:], .1)
