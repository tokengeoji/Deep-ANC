#!/usr/bin/env python3
"""실제 Jetson Docker에서 CUDA·cuDNN·ORT·TensorRT를 무오디오 검증한다.

고정 난수의 작은 Conv1d 모형은 설치 진단용이며 새 ANC 학습모델이 아니다.
오디오 장치·체크포인트를 열지 않고 ONNX/엔진도 메모리에서만 생성한다.
감쇠 성능, 프로젝트 모델의 스트리밍 등가성, 실시간 지연은 검증하지 않는다.

  .venv/bin/python scripts/bench/check_jetson_stack.py
  .venv/bin/python scripts/bench/check_jetson_stack.py --output runs/jetson_stack.json

JSON은 항상 stdout에 출력하며 --output은 기존 파일을 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys


def _compare(np, reference, actual) -> dict:
    if reference.shape != actual.shape:
        raise RuntimeError(f"출력 shape 불일치: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise RuntimeError("출력에 NaN/Inf가 있습니다")
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "atol": 1e-5, "rtol": 1e-4,
        "passed": bool(np.allclose(reference, actual, atol=1e-5, rtol=1e-4)),
    }


def _require_comparison(report: dict, name: str, result: dict) -> None:
    report["checks"][name] = result
    if not result["passed"]:
        raise RuntimeError(f"{name} 수치 오차가 허용 범위를 넘었습니다: {result}")


def _check_stack(report: dict) -> None:
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Docker 컨테이너 내부에서만 실행하세요")
    if platform.machine() != "aarch64":
        raise RuntimeError("실제 ARM64 Jetson에서만 검증할 수 있습니다")
    release_path = Path("/etc/nv_tegra_release")
    report["environment"]["l4t_release"] = release_path.read_text().splitlines()[0]

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    report["versions"].update(
        numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__,
        torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )
    if ort.__version__ != "1.18.1":
        raise RuntimeError("Tegra 규약: onnxruntime==1.18.1이 필요합니다")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch에서 CUDA를 사용할 수 없습니다")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError("PyTorch에서 cuDNN을 사용할 수 없습니다")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report["environment"]["gpu"] = torch.cuda.get_device_name(0)
    report["environment"]["gpu_capability"] = list(torch.cuda.get_device_capability(0))

    with torch.inference_mode():
        a, b = torch.randn(32, 32), torch.randn(32, 32)
        gpu_matmul = a.cuda() @ b.cuda()
        torch.cuda.synchronize()
        _require_comparison(report, "cuda_matmul", _compare(
            np, (a @ b).numpy(), gpu_matmul.cpu().numpy(),
        ))

        model = torch.nn.Sequential(
            torch.nn.Conv1d(4, 8, 3), torch.nn.ReLU(), torch.nn.Conv1d(8, 4, 1),
        ).eval()
        x = torch.randn(1, 4, 256)
        reference = model(x).numpy()
        x_gpu = x.cuda()
        model_gpu = copy.deepcopy(model).cuda()
        # PyTorch Conv1d와 같이 높이 1의 4D 뷰로 cuDNN을 명시적으로 호출한다.
        cudnn_output = torch.ops.aten.cudnn_convolution.default(
            x_gpu.unsqueeze(2), model_gpu[0].weight.unsqueeze(2),
            [0, 0], [1, 1], [1, 1], 1, False, True, False,
        ).squeeze(2) + model_gpu[0].bias.view(1, -1, 1)
        torch.cuda.synchronize()
        _require_comparison(report, "cudnn_conv1d", _compare(
            np, model[0](x).numpy(), cudnn_output.cpu().numpy(),
        ))
        gpu_output = model_gpu(x_gpu)
        torch.cuda.synchronize()
        _require_comparison(report, "torch_cuda_conv1d_model", _compare(
            np, reference, gpu_output.cpu().numpy(),
        ))

        buffer = io.BytesIO()
        torch.onnx.export(
            model, x, buffer, input_names=["x"], output_names=["y"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    onnx_bytes = buffer.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(onnx_bytes), full_check=True)
    report["model"] = {
        "purpose": "설치 검증 전용 고정 난수 Conv1d; 학습·ANC 모델 아님",
        "seed": 0, "input_shape": list(x.shape), "output_shape": list(reference.shape),
        "opset": 17, "dynamic_axes": False, "onnx_bytes": len(onnx_bytes),
        "precision": "FP32; TF32 비활성",
    }

    options = ort.SessionOptions()
    # 명시적인 스레드 수로 Tegra의 기본 affinity 설정을 피한다.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        onnx_bytes, options, providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(["y"], {"x": x.numpy()})[0]
    _require_comparison(report, "ort_cpu", _compare(np, reference, ort_output))
    report["checks"]["ort_cpu"]["providers"] = session.get_providers()

    import tensorrt as trt

    report["versions"]["tensorrt"] = trt.__version__
    if not trt.__version__.startswith("10.3."):
        raise RuntimeError("현재 JetPack 규약의 TensorRT 10.3.x가 필요합니다")

    class DiagnosticLogger(trt.ILogger):
        def log(self, severity, message):
            if severity <= trt.ILogger.Severity.WARNING:
                print(f"TensorRT: {message}", file=sys.stderr)

    logger = DiagnosticLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10은 explicit batch가 기본이다.
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse 실패: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 * 1024 * 1024)
    config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("TensorRT 엔진 역직렬화 실패")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT 실행 컨텍스트 생성 실패")
    if engine.num_io_tensors != 2:
        raise RuntimeError("TensorRT 입력/출력 수가 예상과 다릅니다")
    for name in ("x", "y"):
        if engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f"TensorRT {name} dtype이 FP32가 아닙니다")
    output_shape = tuple(engine.get_tensor_shape("y"))
    if output_shape != reference.shape:
        raise RuntimeError(f"TensorRT 출력 shape 불일치: {output_shape}")
    trt_output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    if not context.set_tensor_address("x", x_gpu.data_ptr()):
        raise RuntimeError("TensorRT 입력 포인터 설정 실패")
    if not context.set_tensor_address("y", trt_output.data_ptr()):
        raise RuntimeError("TensorRT 출력 포인터 설정 실패")
    stream = torch.cuda.current_stream()
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT 실행 실패")
    stream.synchronize()
    _require_comparison(report, "tensorrt_fp32", _compare(
        np, reference, trt_output.cpu().numpy(),
    ))
    report["checks"]["tensorrt_fp32"]["engine_bytes"] = plan.nbytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="새 JSON 파일만 생성 (부모 폴더는 기존 경로)")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"기존 출력 파일을 덮어쓰지 않습니다: {args.output}")
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "scope": "실제 Jetson Docker의 무오디오 합성 설치 검증",
        "not_validated": ["ANC 감쇠", "실제 I/O 지연", "프로젝트 모델 추론", "실시간 마감"],
        "environment": {"architecture": platform.machine(), "python": platform.python_version()},
        "versions": {}, "checks": {},
    }
    try:
        _check_stack(report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(output)
        except OSError as exc:
            print(f"JSON 기록 실패: {exc}", file=sys.stderr)
            sys.stdout.write(output)
            return 1
    sys.stdout.write(output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
