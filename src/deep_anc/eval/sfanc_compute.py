"""무오디오 계산시간 진단. 장치 왕복 지연·실시간 마감·ANC 성능 검증이 아니다."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from deepanc.calibration import DEFAULT_RIR, load_secondary_path
from deep_anc.baselines.sfanc_stream import StreamingFIRBank
from deep_anc.models.hybrid_anc import HybridANCNet
from deep_anc.train.sfanc_experiment import load_selector
from deep_anc.train.sfanc_selector import predict_selector


ROOT = Path(__file__).resolve().parents[3]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing_summary(milliseconds, *, samples):
    """두 표본률의 산술 예산을 병기한다. 모델 표본률을 변경하지 않는다."""
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("측정값은 유한한 비음수 1차원 배열이어야 합니다")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples는 양의 정수여야 합니다")
    result = {name: float(value) for name, value in zip(
        ("median_ms", "p95_ms", "p99_ms", "max_ms"),
        (*np.percentile(values, [50, 95, 99]), values.max()),
    )}
    result["trials"] = len(values)
    result["durations_ms"] = values.tolist()
    result["duration_budget_comparison_only"] = {
        str(rate): {
            "samples": samples,
            "duration_ms": samples * 1000.0 / rate,
            "observed_calls_over_duration": int(np.count_nonzero(values > samples * 1000.0 / rate)),
        } for rate in (16000, 48000)
    }
    return result


def _measure(operation, *, warmup, trials, device):
    """CUDA는 시작 전/결과 반환 후 동기화한다. 결과 유한값 검사는 타이머 밖이다."""
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    durations = []
    for iteration in range(warmup + trials):
        synchronize()
        start = time.perf_counter_ns()
        output = operation()
        synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not np.isfinite(output).all():
            raise ValueError("계산 출력에 비유한 값이 있습니다")
        if iteration >= warmup:
            durations.append(elapsed)
    return durations


def _load_omap(directory):
    directory = Path(directory)
    with torch.random.fork_rng(devices=[]):
        model, coefficients, artifact = load_selector(directory)
    source = json.loads((directory / "report.json").read_text())
    if artifact["sample_rate"] != 16000 or source["config"]["secondary_kind"] != "omap_measured":
        raise ValueError("이 벤치는 OMAP 16 kHz 원본 S 기반 artifact 전용입니다")
    with np.load(directory / "bank.npz", allow_pickle=False) as bank:
        if not np.array_equal(bank["secondary"], load_secondary_path()):
            raise ValueError("bank의 S가 OMAP 원본 전체 탭과 다릅니다")
    if len(coefficients) < 3 or not np.any(coefficients[1:]) or np.any(coefficients[0]):
        raise ValueError("bank는 첫 zero 후보와 최소 두 FIR 후보가 필요합니다")
    return model, coefficients, artifact


def benchmark_compute(run_directory, out, *, hybrid_config=ROOT / "configs/model_tiny.yaml",
                      devices=("cpu",), blocks=(16, 32, 64, 128, 256), warmup=30,
                      trials=200, torch_threads=1, seed=20260920):
    """저장된 SFANC + 고정 seed 미학습 Hybrid 구조의 wall-time을 새 디렉터리에 저장."""
    started_at = datetime.now(timezone.utc).isoformat()
    destination = Path(out)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"기존 출력은 덮어쓰지 않습니다: {destination}")
    for name, value, minimum, maximum in (("warmup", warmup, 0, 10000),
                                         ("trials", trials, 1, 100000),
                                         ("torch_threads", torch_threads, 1, 64),
                                         ("seed", seed, 0, 2**32 - 1)):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} 범위/정수 오류")
    if not blocks or len(set(blocks)) != len(blocks) or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > 4096 for n in blocks
    ):
        raise ValueError("blocks는 중복 없는 1~4096 정수 목록이어야 합니다")
    if not devices or len(set(devices)) != len(devices) or any(d not in ("cpu", "cuda") for d in devices):
        raise ValueError("devices는 cpu/cuda의 중복 없는 목록이어야 합니다")
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("요청한 CUDA를 사용할 수 없습니다. CPU로 대체하지 않습니다")
    selector, coefficients, artifact = _load_omap(run_directory)
    configuration = yaml.safe_load(Path(hybrid_config).read_text())
    if configuration.get("name") != "hybrid_anc_tiny":
        raise ValueError("이 벤치는 기존 HybridANCNet tiny 설정 전용입니다")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        hybrid = HybridANCNet(configuration).eval()
    if hybrid.hop != 128 or hybrid.context > 256 or hybrid.in_channels != 2:
        raise ValueError("Hybrid의 원래 256샘플/2채널 streaming 계약을 유지해야 합니다")
    state_digest = hashlib.sha256()
    for name, tensor in hybrid.state_dict().items():
        state_digest.update(name.encode())
        state_digest.update(tensor.detach().numpy().tobytes())

    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    reference = rng.normal(scale=.02, size=artifact["window_samples"]).astype(np.float32)
    inputs = {size: rng.normal(scale=.02, size=size).astype(np.float64) for size in blocks}
    hybrid_input = np.zeros((1, 2, 256), dtype=np.float32)
    hybrid_input[0, 0] = rng.normal(scale=.02, size=256)
    measurements = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(torch_threads)
        with torch.inference_mode():
            for size in blocks:
                for mode in ("steady", "switch_each_block"):
                    fir = StreamingFIRBank(coefficients, initial_index=1,
                                           transition_samples=size if mode == "switch_each_block" else 0,
                                           control_limit=.2)
                    selected = 1

                    def filter_operation():
                        nonlocal selected
                        if mode == "switch_each_block":
                            selected = 2 if selected == 1 else 1
                            fir.request_filter(selected)
                        return fir.process(inputs[size])

                    durations = _measure(filter_operation, warmup=warmup, trials=trials,
                                         device=torch.device("cpu"))
                    measurements.append({"component": "sfanc_fir", "device": "cpu", "mode": mode,
                        "block_samples": size, "dtype": "float64", "control_limit": .2,
                        "transition_samples": size if mode == "switch_each_block" else 0,
                        "timing_scope": "Python API 전체: 선택 요청(전환 모드), FIR, 상태, limiter, 내부 검사",
                        **timing_summary(durations, samples=size)})
            for device_name in devices:
                device = torch.device(device_name)
                active_selector = copy.deepcopy(selector).to(device).eval()
                durations = _measure(lambda: predict_selector(active_selector, reference,
                                      n_fft=artifact["n_fft"]), warmup=warmup, trials=trials, device=device)
                measurements.append({"component": "sfanc_selector", "device": str(device),
                    "window_samples": artifact["window_samples"], "dtype": "float32",
                    "timing_scope": "CPU Welch 특징 + 전송 + CNN + argmax + CPU 결과 회수 + CUDA 동기화",
                    **timing_summary(durations, samples=artifact["window_samples"])})
                active_hybrid = copy.deepcopy(hybrid).to(device).eval()
                states = active_hybrid.init_states(device=device)

                def hybrid_operation():
                    nonlocal states
                    block = torch.from_numpy(hybrid_input).to(device)
                    output, states = active_hybrid.streaming_step(block, states)
                    return output.cpu().numpy()

                durations = _measure(hybrid_operation, warmup=warmup, trials=trials, device=device)
                measurements.append({"component": "hybrid_anc_tiny_untrained", "device": str(device),
                    "block_samples": 256, "dtype": "float32", "err_input": "zero",
                    "state_retained_between_calls": True,
                    "timing_scope": "CPU 입력에서 전송 + PyTorch streaming_step + CPU 출력 회수 + CUDA 동기화",
                    **timing_summary(durations, samples=256)})
    finally:
        torch.set_num_threads(previous_threads)

    report = {
        "schema": "sfanc_compute_benchmark.v1",
        "started_at_utc": started_at,
        "audio_opened": False, "end_to_end_latency_measured": False,
        "anc_attenuation_compared": False, "real_time_claim_allowed": False,
        "deployment_allowed": False, "components_run_concurrently": False,
        "hybrid_weights": "fixed_seed_untrained_structure_only_no_checkpoint",
        "hybrid_trained_for_omap_16khz": False,
        "fir_implementation": "Python/NumPy 오프라인 수치 코어; production native 실시간 최적화 아님",
        "scope": ["측정은 구성요소를 순차 실행한 유한 횟수 wall-time이며 최악 실행시간 증명이 아님",
                  "I/O·블록 수집·스레드 handoff·스피커·음향 경로 지연은 미포함",
                  "selector 관측창 길이는 선택 반응시간이며 매 샘플 출력 지연으로 더하지 않음",
                  "48/16 kHz 예산은 샘플 수/표본률 산술 비교; 모델 재학습·리샘플링·배포 승인 아님",
                  "같은 합성 입력 블록을 반복하며 Hybrid ERR은 0; closed-loop 또는 음원 평가 아님",
                  "torch eager FP32이며 ONNX/TensorRT·native FIR 최적화본과의 우위 비교 아님"],
        "settings": {"warmup": warmup, "trials": trials, "torch_threads": torch_threads,
                     "seed": seed, "devices": list(devices), "blocks": list(blocks)},
        "environment": {"machine": platform.machine(), "kernel": platform.release(),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_device": torch.cuda.get_device_name() if "cuda" in devices else None,
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "source": {"sfanc_run": str(Path(run_directory).resolve()),
            "sfanc_sample_rate_hz": artifact["sample_rate"], "bank_shape": list(coefficients.shape),
            "artifacts_sha256": {name: _sha(Path(run_directory) / name)
                                 for name in ("bank.npz", "selector.pt", "report.json")},
            "omap_rir_sha256": _sha(DEFAULT_RIR), "omap_secondary_exact": True,
            "hybrid_config": str(Path(hybrid_config).resolve()), "hybrid_config_sha256": _sha(hybrid_config),
            "hybrid_initial_state_sha256": state_digest.hexdigest(),
            "hybrid_parameters": sum(p.numel() for p in hybrid.parameters()),
            "implementation_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in (
                Path(__file__), ROOT / "src/deep_anc/baselines/sfanc_stream.py",
                ROOT / "src/deep_anc/train/sfanc_selector.py", ROOT / "src/deep_anc/models/hybrid_anc.py")}},
        "measurements": measurements,
    }
    with (destination / "report.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report
