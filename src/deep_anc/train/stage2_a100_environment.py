"""Stage-2 production A100 runtime의 실제 visible-device/결정론 계약.

``nvidia-smi``의 parent GPU inventory만으로는 MIG slice 또는 40 GB 장치를
배제할 수 없다. 이 모듈은 typed campaign admission이 끝난 뒤, CUDA tensor를
만들기 전에 호출되어 *현재 PyTorch process에 보이는* 정확히 한 장의 80 GB
A100과 deterministic backend 상태를 봉인한다.

환경변수 ``CUBLAS_WORKSPACE_CONFIG``는 CUDA context 생성 뒤 설정해도 효력이
보장되지 않는다. 따라서 이 모듈은 값을 대신 채우지 않고 process 시작 시점의
``:4096:8``을 요구한다. 나머지 PyTorch/cuDNN flag는 여기서 설정한 뒤 다시
읽어 exact receipt로 만든다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping


STAGE2_A100_ENVIRONMENT_SCHEMA = "stage2_2khz_a100_environment_v1"
STAGE2_A100_REQUIRED_TORCH_VERSION = "2.5.1+cu121"
STAGE2_A100_REQUIRED_CUDA_VERSION = "12.1"
STAGE2_A100_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES = 79 * 1024**3
STAGE2_A100_MAX_VISIBLE_MEMORY_BYTES = 82 * 1024**3

_A100_80GB_NAME = re.compile(r"A100.*80\s*GB", re.IGNORECASE)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stage2_a100_environment_sha256(payload: Mapping[str, Any]) -> str:
    """receipt 자체의 content identity를 계산한다."""

    return hashlib.sha256(_canonical_json(dict(payload))).hexdigest()


def validate_stage2_a100_environment_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """persisted/live A100 payload를 pure validation한다.

    테스트에서는 이 함수를 사용해 40 GB A100과 MIG slice를 실제 CUDA 접근 없이
    거부한다.
    """

    expected = {
        "schema",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "cuda_initialized_before_configuration",
        "cuda_initialized_after_probe",
        "device_count",
        "current_device",
        "device_name",
        "visible_total_memory_bytes",
        "bf16_supported",
        "world_size",
        "rank",
        "local_rank",
        "cuda_visible_devices",
        "nvidia_smi_l_sha256",
        "mig_detected",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "matmul_allow_tf32",
    }
    value = dict(payload)
    if set(value) != expected:
        raise ValueError("Stage-2 A100 environment receipt key 집합이 exact하지 않습니다")
    if value["schema"] != STAGE2_A100_ENVIRONMENT_SCHEMA:
        raise ValueError("Stage-2 A100 environment schema가 다릅니다")
    if value["torch_version"] != STAGE2_A100_REQUIRED_TORCH_VERSION:
        raise ValueError("Stage-2 A100 torch version이 exact하지 않습니다")
    if value["torch_cuda_version"] != STAGE2_A100_REQUIRED_CUDA_VERSION:
        raise ValueError("Stage-2 A100 CUDA runtime version이 exact하지 않습니다")
    if value["cuda_available"] is not True:
        raise ValueError("Stage-2 A100 CUDA가 available이 아닙니다")
    if value["cuda_initialized_before_configuration"] is not False:
        raise ValueError("Stage-2 결정론 설정 전에 CUDA context가 이미 생성됐습니다")
    if value["cuda_initialized_after_probe"] is not True:
        raise ValueError("Stage-2 A100 probe가 실제 visible CUDA device를 열지 못했습니다")
    if type(value["device_count"]) is not int or value["device_count"] != 1:
        raise ValueError("Stage-2 canonical pretrain은 PyTorch visible device가 정확히 1개여야 합니다")
    if type(value["current_device"]) is not int or value["current_device"] != 0:
        raise ValueError("Stage-2 canonical pretrain current CUDA device는 0이어야 합니다")
    name = str(value["device_name"])
    if not _A100_80GB_NAME.search(name) or "MIG" in name.upper():
        raise ValueError(f"Stage-2 visible GPU가 full A100 80GB가 아닙니다: {name!r}")
    memory = value["visible_total_memory_bytes"]
    if (
        type(memory) is not int
        or not STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES
        <= memory
        <= STAGE2_A100_MAX_VISIBLE_MEMORY_BYTES
    ):
        raise ValueError(
            "Stage-2 PyTorch visible memory가 full A100 80GB 범위가 아닙니다: "
            f"{memory!r}"
        )
    if value["bf16_supported"] is not True:
        raise ValueError("Stage-2 visible GPU가 bf16을 지원하지 않습니다")
    for key, expected_value in (
        ("world_size", 1),
        ("rank", 0),
        ("local_rank", 0),
    ):
        if type(value[key]) is not int or value[key] != expected_value:
            raise ValueError(f"Stage-2 {key}가 exact world1 process 계약과 다릅니다")
    if value["mig_detected"] is not False:
        raise ValueError("Stage-2 canonical pretrain에 MIG partition은 허용하지 않습니다")
    smi_digest = str(value["nvidia_smi_l_sha256"])
    if len(smi_digest) != 64 or any(ch not in "0123456789abcdef" for ch in smi_digest):
        raise ValueError("Stage-2 nvidia-smi -L snapshot SHA가 없습니다")
    if value["cublas_workspace_config"] != STAGE2_A100_REQUIRED_CUBLAS_WORKSPACE:
        raise ValueError("Stage-2 CUBLAS_WORKSPACE_CONFIG가 :4096:8이 아닙니다")
    exact_flags = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    }
    for key, expected_value in exact_flags.items():
        if value[key] is not expected_value:
            raise ValueError(f"Stage-2 결정론 backend flag {key}가 다릅니다")
    visible = value["cuda_visible_devices"]
    if visible is not None and (not isinstance(visible, str) or not visible.strip()):
        raise ValueError("CUDA_VISIBLE_DEVICES는 None 또는 nonempty string이어야 합니다")
    return value


def configure_and_collect_stage2_a100_environment(
    torch_module: Any,
    *,
    nvidia_smi_l_output: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """CUDA tensor 생성 전에 production A100 환경을 설정·검증한다."""

    environment = os.environ if environ is None else environ
    if str(getattr(torch_module, "__version__", "")) != STAGE2_A100_REQUIRED_TORCH_VERSION:
        raise RuntimeError(
            "Stage-2 canonical pretrain은 torch 2.5.1+cu121 exact가 필요합니다"
        )
    if str(getattr(getattr(torch_module, "version", None), "cuda", "")) != (
        STAGE2_A100_REQUIRED_CUDA_VERSION
    ):
        raise RuntimeError("Stage-2 canonical pretrain은 torch CUDA 12.1 exact가 필요합니다")
    cuda = torch_module.cuda
    initialized_before = bool(cuda.is_initialized())
    if initialized_before:
        raise RuntimeError("Stage-2 A100 결정론 설정 전에 CUDA context가 이미 생성됐습니다")
    workspace = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace != STAGE2_A100_REQUIRED_CUBLAS_WORKSPACE:
        raise RuntimeError(
            "Stage-2 실행 process를 CUBLAS_WORKSPACE_CONFIG=:4096:8로 시작해야 합니다"
        )
    world_size = int(environment.get("WORLD_SIZE", "1"))
    rank = int(environment.get("RANK", "0"))
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    if (world_size, rank, local_rank) != (1, 0, 0):
        raise RuntimeError("Stage-2 canonical pretrain은 exact world_size=1/rank=0/local_rank=0입니다")
    if not isinstance(nvidia_smi_l_output, str) or not nvidia_smi_l_output.strip():
        raise RuntimeError("Stage-2 A100 nvidia-smi -L snapshot이 비었습니다")
    upper_inventory = nvidia_smi_l_output.upper()
    mig_detected = "MIG" in upper_inventory
    if mig_detected:
        raise RuntimeError("Stage-2 canonical pretrain은 MIG inventory를 허용하지 않습니다")

    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False

    if not bool(cuda.is_available()):
        raise RuntimeError("Stage-2 canonical pretrain CUDA GPU가 없습니다")
    if int(cuda.device_count()) != 1:
        raise RuntimeError("Stage-2 canonical pretrain은 PyTorch visible GPU가 정확히 1개여야 합니다")
    current = int(cuda.current_device())
    properties = cuda.get_device_properties(current)
    payload = {
        "schema": STAGE2_A100_ENVIRONMENT_SCHEMA,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "cuda_available": True,
        "cuda_initialized_before_configuration": initialized_before,
        "cuda_initialized_after_probe": bool(cuda.is_initialized()),
        "device_count": int(cuda.device_count()),
        "current_device": current,
        "device_name": str(properties.name),
        "visible_total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(cuda.is_bf16_supported()),
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_l_sha256": hashlib.sha256(
            nvidia_smi_l_output.encode("utf-8")
        ).hexdigest(),
        "mig_detected": mig_detected,
        "cublas_workspace_config": workspace,
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
    }
    return validate_stage2_a100_environment_payload(payload)


__all__ = [
    "STAGE2_A100_ENVIRONMENT_SCHEMA",
    "STAGE2_A100_MAX_VISIBLE_MEMORY_BYTES",
    "STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES",
    "STAGE2_A100_REQUIRED_CUBLAS_WORKSPACE",
    "STAGE2_A100_REQUIRED_CUDA_VERSION",
    "STAGE2_A100_REQUIRED_TORCH_VERSION",
    "configure_and_collect_stage2_a100_environment",
    "stage2_a100_environment_sha256",
    "validate_stage2_a100_environment_payload",
]
