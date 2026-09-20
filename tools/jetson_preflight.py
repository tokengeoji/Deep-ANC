#!/usr/bin/env python3
"""Inspect a Jetson environment without changing packages or device settings."""

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_GUIDE = (
    "https://docs.nvidia.com/deeplearning/frameworks/"
    "install-pytorch-jetson-platform/index.html"
)
PYTORCH_MATRIX = (
    "https://docs.nvidia.com/deeplearning/frameworks/"
    "install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html"
)


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").replace("\x00", "").strip()
    except OSError:
        return None


def memory_info():
    """Report host memory; Jetson GPU memory is shared with the host."""
    content = read_text("/proc/meminfo")
    result = {}
    if content:
        for line in content.splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                result[key + "_bytes"] = int(value.strip().split()[0]) * 1024
    return result


def jetpack_version():
    if shutil.which("dpkg-query") is None:
        return None
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def dependency_info():
    result = {}
    for name in ("numpy", "scipy", "soundfile"):
        try:
            module = importlib.import_module(name)
            result[name] = {"ok": True, "version": str(module.__version__)}
        except Exception as exc:
            result[name] = {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
    return result


def required_torch_apis(torch_module):
    """Check vendor builds by capabilities, not by parsing their version suffix."""
    fft = getattr(torch_module, "fft", None)
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch_module.load).parameters
    except (AttributeError, TypeError, ValueError):
        supports_weights_only = False
    return {
        "torch.fft.rfft": callable(getattr(fft, "rfft", None)),
        "torch.fft.irfft": callable(getattr(fft, "irfft", None)),
        "torch.load(weights_only=...)": supports_weights_only,
    }


def torch_info():
    result = {"import_ok": False, "cuda_available": False, "cuda_compute_ok": False}
    try:
        import torch

        result.update(
            import_ok=True,
            version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            module_path=str(torch.__file__),
            required_apis=required_torch_apis(torch),
        )
        result["cuda_available"] = bool(torch.cuda.is_available())
        if not result["cuda_available"]:
            return result
        result["device_count"] = torch.cuda.device_count()
        result["device_name"] = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        result["device_memory_bytes"] = properties.total_memory
        result["compute_capability"] = [properties.major, properties.minor]
        # Allocation, GPU kernels, and autograd must all succeed. Availability
        # alone does not catch an incompatible wheel or missing CUDA kernels.
        sample = torch.arange(256, dtype=torch.float32, device="cuda", requires_grad=True)
        loss = torch.tanh(sample / 256.0).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(loss).item()) and bool(torch.isfinite(sample.grad).all().item())
        result["cuda_compute_ok"] = finite
        result["cuda_probe_loss"] = float(loss.item())
        if not finite:
            result["error"] = "CUDA forward/backward produced non-finite values."
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    return result


def evaluate_report(report, require_cuda=False, require_jetson=False):
    """Return actionable failures and warnings separately."""
    errors = []
    warnings = []
    if tuple(report["python"]["version_info"]) < (3, 8):
        errors.append("Python 3.8 or newer is required.")
    if require_jetson and not report["system"]["is_jetson"]:
        errors.append(
            "Jetson hardware was not detected. Run on the Orin host; for local "
            "diagnostics use --allow-cpu without --require-jetson."
        )
    for name, dependency in report["dependencies"].items():
        if not dependency["ok"]:
            errors.append(
                "{} cannot be imported: {}. Install requirements-jetson.txt "
                "with the selected Python interpreter.".format(name, dependency["error"])
            )
    pytorch = report["torch"]
    if not pytorch["import_ok"]:
        errors.append(
            "PyTorch cannot be imported: {}. Preserve or install the NVIDIA build "
            "matching this JetPack release; see {} and {}.".format(
                pytorch.get("error", "unknown import failure"), PYTORCH_GUIDE, PYTORCH_MATRIX
            )
        )
    elif require_cuda and not pytorch["cuda_compute_ok"]:
        errors.append(
            "CUDA forward/backward validation failed: {}. Check the JetPack/PyTorch "
            "compatibility matrix {}. A CPU-only torch wheel cannot prepare GPU training.".format(
                pytorch.get("error", "torch.cuda.is_available() is false"), PYTORCH_MATRIX
            )
        )
    elif not pytorch["cuda_compute_ok"]:
        warnings.append("CUDA was not validated; this report permits CPU diagnostics only.")
    if pytorch["import_ok"]:
        missing_apis = [name for name, available in pytorch["required_apis"].items() if not available]
        if missing_apis:
            errors.append(
                "Required PyTorch APIs are missing: {}. Training/checkpoint resume requires "
                "these APIs (upstream PyTorch 1.13+); select a JetPack-compatible NVIDIA "
                "build using {}. No packages were upgraded.".format(
                    ", ".join(missing_apis), PYTORCH_MATRIX
                )
            )
    if report["storage"]["free_bytes"] < 5 * 1024 ** 3:
        warnings.append("Less than 5 GiB free on the repository filesystem; plan space for recordings and checkpoints.")
    available = report["memory"].get("MemAvailable_bytes")
    if available is not None and available < 2 * 1024 ** 3:
        warnings.append("Less than 2 GiB host memory available; close unused applications before training.")
    if report["system"]["is_jetson"] and report["system"]["jetpack_package_version"] is None:
        warnings.append("nvidia-jetpack metapackage was not found; identify JetPack from the recorded L4T release.")
    return errors, warnings


def collect_report():
    architecture = platform.machine().lower()
    model = read_text("/proc/device-tree/model") or read_text("/sys/firmware/devicetree/base/model")
    l4t = read_text("/etc/nv_tegra_release")
    disk = shutil.disk_usage(str(REPO_ROOT))
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "system": {
            "architecture": architecture,
            "platform": platform.platform(),
            "model": model,
            "l4t_release": l4t,
            "jetpack_package_version": jetpack_version(),
            "is_jetson": architecture in ("aarch64", "arm64")
            and ("jetson" in (model or "").lower() or l4t is not None),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:2]),
            "virtual_environment": sys.prefix != sys.base_prefix,
        },
        "dependencies": dependency_info(),
        "torch": torch_info(),
        "memory": memory_info(),
        "storage": {"total_bytes": disk.total, "free_bytes": disk.free},
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--require-cuda", action="store_true", help="Fail unless CUDA forward/backward succeeds.")
    device.add_argument("--allow-cpu", action="store_true", help="Explicitly permit local CPU diagnostics.")
    parser.add_argument("--require-jetson", action="store_true", help="Fail unless a Jetson/L4T host is detected.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "jetson_preflight.json")
    args = parser.parse_args(argv)
    report = collect_report()
    errors, warnings = evaluate_report(report, require_cuda=args.require_cuda, require_jetson=args.require_jetson)
    report.update(
        ok=not errors,
        requirements={"cuda": args.require_cuda, "jetson": args.require_jetson},
        errors=errors,
        warnings=warnings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Preflight: {} | Python {} | {} | CUDA compute {}".format(
        "PASS" if report["ok"] else "FAIL",
        report["python"]["version"],
        report["system"]["model"] or report["system"]["architecture"],
        "PASS" if report["torch"]["cuda_compute_ok"] else "NOT VALIDATED",
    ))
    print("Report: {}".format(args.output.resolve()))
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    for error in errors:
        print("ERROR: " + error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
