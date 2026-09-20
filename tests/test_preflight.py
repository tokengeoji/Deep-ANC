"""Readiness checks must fail closed when the requested hardware is unavailable."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "jetson_preflight.py"
SPEC = importlib.util.spec_from_file_location("jetson_preflight", MODULE_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def healthy_report():
    return {
        "python": {"version_info": [3, 8]},
        "system": {"is_jetson": True, "jetpack_package_version": "test"},
        "dependencies": {"numpy": {"ok": True}},
        "torch": {"import_ok": True, "cuda_available": True, "cuda_compute_ok": True,
                  "required_apis": {"torch.fft.rfft": True, "torch.fft.irfft": True,
                                    "torch.load(weights_only=...)": True}},
        "storage": {"free_bytes": 10 * 1024 ** 3},
        "memory": {"MemAvailable_bytes": 4 * 1024 ** 3},
    }


class PreflightRequirementsTest(unittest.TestCase):
    def test_compatible_jetson_passes(self):
        errors, warnings = PREFLIGHT.evaluate_report(healthy_report(), require_cuda=True, require_jetson=True)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_cuda_availability_without_working_kernels_fails(self):
        report = healthy_report()
        report["torch"]["cuda_compute_ok"] = False
        report["torch"]["error"] = "no kernel image is available"
        errors, _ = PREFLIGHT.evaluate_report(report, require_cuda=True)
        self.assertTrue(any("CUDA forward/backward" in error for error in errors))
        self.assertTrue(any("no kernel image" in error for error in errors))

    def test_non_jetson_gpu_is_not_jetson_ready(self):
        report = healthy_report()
        report["system"]["is_jetson"] = False
        errors, _ = PREFLIGHT.evaluate_report(report, require_cuda=True, require_jetson=True)
        self.assertTrue(any("Jetson hardware" in error for error in errors))

    def test_cpu_diagnostics_permit_no_gpu(self):
        report = healthy_report()
        report["system"]["is_jetson"] = False
        report["torch"]["cuda_available"] = False
        report["torch"]["cuda_compute_ok"] = False
        errors, warnings = PREFLIGHT.evaluate_report(report)
        self.assertEqual(errors, [])
        self.assertTrue(any("CPU diagnostics" in warning for warning in warnings))

    def test_missing_pytorch_does_not_pass_cpu_preparation(self):
        report = healthy_report()
        report["torch"] = {"import_ok": False, "cuda_available": False, "cuda_compute_ok": False}
        errors, _ = PREFLIGHT.evaluate_report(report)
        self.assertTrue(any("PyTorch cannot be imported" in error for error in errors))

    def test_dependency_import_failure_is_actionable(self):
        report = healthy_report()
        report["dependencies"]["soundfile"] = {"ok": False, "error": "libsndfile missing"}
        original = copy.deepcopy(report)
        errors, _ = PREFLIGHT.evaluate_report(report)
        self.assertTrue(any("libsndfile missing" in error for error in errors))
        self.assertEqual(report, original)

    def test_working_cuda_does_not_hide_missing_checkpoint_api(self):
        report = healthy_report()
        report["torch"]["required_apis"]["torch.load(weights_only=...)"] = False
        errors, _ = PREFLIGHT.evaluate_report(report, require_cuda=True)
        self.assertTrue(any("weights_only" in error and "JetPack-compatible" in error for error in errors))

    def test_legacy_load_kwargs_are_not_weights_only_support(self):
        def legacy_load(path, **pickle_load_args):
            pass

        module = SimpleNamespace(fft=SimpleNamespace(rfft=lambda x: x, irfft=lambda x: x), load=legacy_load)
        detected = PREFLIGHT.required_torch_apis(module)
        self.assertTrue(detected["torch.fft.rfft"])
        self.assertTrue(detected["torch.fft.irfft"])
        self.assertFalse(detected["torch.load(weights_only=...)"])

    def test_missing_fft_api_is_reported(self):
        def supported_load(path, *, weights_only=False):
            pass

        module = SimpleNamespace(load=supported_load)
        detected = PREFLIGHT.required_torch_apis(module)
        self.assertFalse(detected["torch.fft.rfft"])
        self.assertFalse(detected["torch.fft.irfft"])
        self.assertTrue(detected["torch.load(weights_only=...)"])
        report = healthy_report()
        report["torch"]["required_apis"] = detected
        errors, _ = PREFLIGHT.evaluate_report(report)
        self.assertTrue(any("torch.fft.rfft" in error and "torch.fft.irfft" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
