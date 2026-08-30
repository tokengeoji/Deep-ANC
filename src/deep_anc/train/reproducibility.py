"""재현성 — 시드 고정 + 실행 스냅샷(설정/git hash/pip freeze) 기록."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from .evaluation_contract import snapshot_regular_file


REPRODUCIBILITY_FILES = (
    "config_snapshot.yaml",
    "git_rev.txt",
    "pip_freeze.txt",
    "environment.json",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _publish_or_validate(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists() or path.is_symlink():
        # ``Path.read_bytes``는 symlink를 따라가므로, 공격자가 run 디렉터리의
        # receipt 이름을 외부 파일로 바꿔도 canonical snapshot으로 인정하게 된다.
        # 동일 FD snapshot API로 leaf symlink/retarget/동시 변경을 모두 거부한다.
        existing = snapshot_regular_file(path)
        if existing.content != content:
            raise ValueError(f"canonical reproducibility artifact가 현재 환경과 다릅니다: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def _environment_payload() -> dict:
    payload = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "device_count": int(torch.cuda.device_count()),
        "devices": [],
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        payload["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return payload


def snapshot_run(
    run_dir: str | Path, cfg: dict, *, strict: bool = False
) -> dict[str, str]:
    """resolved config/git/dependencies/device를 기록하고 canonical이면 실패 폐쇄한다."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = yaml.safe_dump(
        cfg, allow_unicode=True, sort_keys=False
    ).encode("utf-8")
    outputs: dict[str, bytes] = {"config_snapshot.yaml": config_bytes}
    commands = (
        ("git_rev.txt", ["git", "rev-parse", "HEAD"]),
        ("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
    )
    for name, cmd in commands:
        try:
            completed = subprocess.run(
                cmd, capture_output=True, timeout=60, check=strict,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            if strict and completed.returncode != 0:
                raise RuntimeError(f"{name} 명령이 실패했습니다: {completed.stderr!r}")
            outputs[name] = bytes(completed.stdout)
        except Exception:
            if strict:
                raise
    outputs["environment.json"] = (
        json.dumps(_environment_payload(), ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    receipts: dict[str, str] = {}
    for name, content in outputs.items():
        if strict:
            receipts[name] = _publish_or_validate(run_dir / name, content)
        else:
            try:
                (run_dir / name).write_bytes(content)
                receipts[name] = hashlib.sha256(content).hexdigest()
            except Exception:
                continue
    if strict and set(receipts) != set(REPRODUCIBILITY_FILES):
        raise RuntimeError("canonical reproducibility receipt가 불완전합니다")
    return receipts
