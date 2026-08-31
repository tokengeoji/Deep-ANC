#!/usr/bin/env python3
"""기존 Elice 인스턴스의 Stage-2 학습 재사용 가능성을 fail-closed 감사한다.

환경을 설치·업데이트하거나 데이터를 다운로드하지 않는다. exact clean commit,
A100 80GB, filesystem, torch/CUDA/bf16, typed Stage-2 campaign의 실제 P/S·public source
bytes를 다시 검사한다. 기존 run은 자동 재개하지 않으며 ``--resume``이 없으면 stale
run/checkpoint를 학습 권한으로 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.stage2_2khz_campaign import (  # noqa: E402
    audit_stage2_2khz_campaign,
)
from deep_anc.train.stage2_a100_environment import (  # noqa: E402
    configure_and_collect_stage2_a100_environment,
    stage2_a100_environment_sha256,
)


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _run(*args: str) -> str:
    environment = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    try:
        return subprocess.run(
            list(args),
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"명령 실행 실패: {' '.join(args)}") from exc


def _relative_file(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    root = REPO_ROOT.resolve(strict=True)
    try:
        path = (root / candidate).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label}가 repository 내부 파일이 아닙니다") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}는 regular non-symlink file이어야 합니다")
    return path


def _gpu_inventory() -> list[dict[str, Any]]:
    raw = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    )
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        name, separator, memory = line.rpartition(",")
        if not separator:
            continue
        try:
            memory_mib = int(memory.strip())
        except ValueError:
            continue
        rows.append({"name": name.strip(), "memory_total_mib": memory_mib})
    if not rows:
        raise RuntimeError("nvidia-smi parent GPU inventory가 비었습니다")
    return rows


def _environment_probe(*, nvidia_smi_l_output: str) -> dict[str, Any]:
    import torch

    return configure_and_collect_stage2_a100_environment(
        torch, nvidia_smi_l_output=nvidia_smi_l_output
    )


def audit_existing_instance(
    *,
    expected_commit: str,
    campaign_path: Path,
    resume_path: Path | None,
    minimum_free_gib: int,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("--expected-commit은 lowercase 전체 40자리 SHA여야 합니다")
    head = _run("git", "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(f"checkout HEAD 불일치: expected={expected_commit}, actual={head}")
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"checkout이 dirty입니다: {status.splitlines()[0]}")

    usage = shutil.disk_usage(REPO_ROOT)
    minimum_total = 128 * 1024**3 - 128 * 1024**2
    minimum_free = int(minimum_free_gib) * 1024**3
    if usage.total < minimum_total or usage.free < minimum_free:
        raise RuntimeError(
            "Stage-2 filesystem 계약 불충족: "
            f"total={usage.total}, free={usage.free}, required_free={minimum_free}"
        )
    payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 campaign 최상위는 mapping이어야 합니다")
    campaign = audit_stage2_2khz_campaign(payload, repo_root=REPO_ROOT)
    if campaign.get("status") != "READY_PRETRAIN":
        raise RuntimeError(
            f"typed Stage-2 campaign이 READY_PRETRAIN이 아닙니다: {campaign.get('status')}"
        )
    # typed artifact chain이 먼저 READY가 된 뒤에만 nvidia-smi/torch CUDA probe를
    # 실행한다. null/missing artifact 기본 상태는 GPU context 전에 끝난다.
    nvidia_smi_l = _run("nvidia-smi", "-L")
    gpu = _gpu_inventory()
    environment = _environment_probe(nvidia_smi_l_output=nvidia_smi_l)
    external_ref = payload["external_contracts"]["canonical_pretrain"]
    external_sha = str(external_ref["sha256"])
    expected_run_dir = (
        REPO_ROOT
        / "runs"
        / f"stage2_pretrain_{external_sha[:12]}_20260803"
    )
    if expected_run_dir.exists():
        if resume_path is None:
            raise RuntimeError(
                "기존 Stage-2 run이 있습니다. 자동 재개하지 않으며 exact --resume이 필요합니다"
            )
        if resume_path.parent != expected_run_dir:
            raise RuntimeError("--resume이 current external-contract run directory 밖입니다")
    elif resume_path is not None:
        raise RuntimeError("새 scratch run에 --resume을 지정할 수 없습니다")

    return {
        "schema": "stage2_2khz_elice_existing_instance_audit_v2",
        "status": "READY_PRETRAIN",
        "expected_commit": expected_commit,
        "actual_commit": head,
        "clean_checkout": True,
        "gpu_inventory": gpu,
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "minimum_free_gib": int(minimum_free_gib),
        "environment": environment,
        "environment_sha256": stage2_a100_environment_sha256(environment),
        "pytorch_visible_full_a100_80gb_verified": True,
        "mig_partition_allowed": False,
        "campaign_preflight_sha256": hashlib.sha256(
            json.dumps(
                campaign,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "all_declared_source_bytes_rehashed": True,
        "stale_run_or_checkpoint_auto_resume_allowed": False,
        "scratch_new_run_directory_required": resume_path is None,
        "explicit_resume": None
        if resume_path is None
        else resume_path.relative_to(REPO_ROOT).as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--campaign", default="configs/stage2_2khz_campaign.yaml"
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--minimum-free-gib", type=int, default=16)
    parser.add_argument("--out", default=None, help="선택적 no-replace JSON")
    args = parser.parse_args(argv)
    try:
        if not 1 <= int(args.minimum_free_gib) <= 128:
            raise ValueError("--minimum-free-gib는 1..128이어야 합니다")
        campaign_path = _relative_file(args.campaign, label="campaign")
        resume = (
            None
            if args.resume is None
            else _relative_file(args.resume, label="resume")
        )
        report = audit_existing_instance(
            expected_commit=args.expected_commit,
            campaign_path=campaign_path,
            resume_path=resume,
            minimum_free_gib=int(args.minimum_free_gib),
        )
        encoded = (
            json.dumps(
                report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        if args.out is not None:
            output = _relative_file(args.out, label="out") if Path(args.out).exists() else REPO_ROOT / args.out
            output.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True))
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
