#!/usr/bin/env python3
"""A100에서 canonical-pretrain exact-resume smoke를 안전하게 실행한다.

이 runner는 canonical ``runs/``를 전혀 사용하지 않는다. 동일한 resolved learning
semantics를 가진 두 arm을

* uninterrupted: 0 → final_step
* resumed: 0 → stop_step, immutable ``stop.pt`` resume → final_step

로 prerequisite root 아래에서 실행하고, checkpoint/telemetry/environment bytes를 묶은
receipt만 만든다. 이 receipt는 campaign ledger의 한 칸일 뿐 init checkpoint나 canonical
completion receipt가 아니다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_train_config  # noqa: E402
from deep_anc.train.a100_pretrain_smoke import (  # noqa: E402
    A100_PRETRAIN_SMOKE_ROLE,
    A100_REQUIRED_CUDA_VERSION,
    A100_REQUIRED_TORCH_VERSION,
    build_a100_pretrain_smoke_artifacts,
    build_a100_pretrain_smoke_environment_receipt,
    build_a100_pretrain_smoke_resume_input,
    finalize_a100_pretrain_smoke_receipt,
    smoke_run_directory,
    validate_a100_pretrain_smoke_receipt,
)
from deep_anc.train.evaluation_contract import (  # noqa: E402
    snapshot_regular_file,
    write_json_exclusive,
)


def _sha_ref(path: Path) -> dict[str, str]:
    snapshot = snapshot_regular_file(path)
    try:
        relative = snapshot.path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:  # pragma: no cover - runner invariant guard
        raise ValueError(f"smoke artifact가 저장소 밖에 있습니다: {snapshot.path}") from exc
    return {"path": relative, "sha256": snapshot.sha256}


def _overrides(
    *,
    bootstrap_sha256: str,
    label: str,
    run_until_step: int,
) -> list[str]:
    return [
        f"experiment_role={A100_PRETRAIN_SMOKE_ROLE}",
        "init_eligible=false",
        "contract_run_dir=false",
        "campaign_prerequisite=null",
        "campaign_prerequisite_sha256=null",
        "init_ckpt=null",
        f"a100_smoke_run_label={label}",
        f"run_until_step={int(run_until_step)}",
        f"data.bootstrap_receipt_sha256={bootstrap_sha256}",
    ]


def _resolved_cfg(
    config: Path,
    *,
    bootstrap_sha256: str,
    label: str,
    run_until_step: int,
) -> dict:
    return load_train_config(
        config,
        _overrides(
            bootstrap_sha256=bootstrap_sha256,
            label=label,
            run_until_step=run_until_step,
        ),
    )


def _run_train(
    config: Path,
    *,
    bootstrap_sha256: str,
    label: str,
    run_until_step: int,
    cublas_workspace_config: str,
    resume: Path | None = None,
) -> dict:
    cfg = _resolved_cfg(
        config,
        bootstrap_sha256=bootstrap_sha256,
        label=label,
        run_until_step=run_until_step,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "train.py"),
        "--config",
        str(config),
    ]
    for override in _overrides(
        bootstrap_sha256=bootstrap_sha256,
        label=label,
        run_until_step=run_until_step,
    ):
        command.extend(["--set", override])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    return cfg


def _link_stop_checkpoint(source: Path, destination: Path) -> None:
    snapshot_regular_file(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"기존 smoke stop checkpoint를 덮어쓸 수 없습니다: {destination}")
    os.link(source, destination, follow_symlinks=False)
    if _sha_ref(source)["sha256"] != _sha_ref(destination)["sha256"]:
        raise RuntimeError("hard-link한 smoke stop checkpoint bytes가 다릅니다")


def _phase_path(run_dir: Path, start_step: int, completed_step: int) -> Path:
    return run_dir / "telemetry" / f"{start_step:06d}_{completed_step:06d}.json"


def _preflight_a100(*, cublas_workspace_config: str) -> None:
    """세 arm을 시작하기 전에 실제 A100/world1 조건을 fail-closed로 확인한다."""

    import torch

    if str(torch.__version__) != A100_REQUIRED_TORCH_VERSION or str(
        torch.version.cuda
    ) != A100_REQUIRED_CUDA_VERSION:
        raise RuntimeError(
            "A100 smoke는 bootstrap과 동일한 torch/CUDA 환경에서만 실행할 수 있습니다: "
            f"torch={torch.__version__!s}, cuda={torch.version.cuda!s}, "
            f"required={A100_REQUIRED_TORCH_VERSION}/{A100_REQUIRED_CUDA_VERSION}"
        )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 1:
        raise RuntimeError(f"A100 smoke는 WORLD_SIZE=1만 허용합니다: {world}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A100 smoke는 CUDA device 1개가 보이는 환경에서만 실행할 수 있습니다")
    name = str(torch.cuda.get_device_name(0))
    if "A100" not in name:
        raise RuntimeError(f"A100 smoke는 A100에서만 실행할 수 있습니다: {name}")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if total_memory < 80 * 1024**3:
        raise RuntimeError(
            "A100 smoke는 80 GiB 이상 GPU가 필요합니다: "
            f"{total_memory / 1024**3:.1f} GiB"
        )
    if cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError("A100 smoke CUBLAS deterministic workspace 값이 승인 목록에 없습니다")
    print(
        "[a100 smoke] preflight PASS — "
        f"device={name}, VRAM={total_memory / 1024**3:.1f}GiB, world={world}, "
        f"CUBLAS_WORKSPACE_CONFIG={cublas_workspace_config}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train_pretrain_tiny.yaml",
        help="canonical tiny pretrain config (기본값: configs/train_pretrain_tiny.yaml)",
    )
    parser.add_argument(
        "--bootstrap-receipt-sha256",
        required=True,
        help="Elice bootstrap receipt의 검증된 SHA-256",
    )
    parser.add_argument("--stop-step", type=int, default=300)
    parser.add_argument("--final-step", type=int, default=500)
    parser.add_argument(
        "--cublas-workspace-config",
        choices=(":4096:8", ":16:8"),
        default=":4096:8",
        help="smoke subprocess에만 주입할 deterministic GEMM 환경값",
    )
    args = parser.parse_args()

    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = (REPO_ROOT / config).resolve()
    bootstrap_sha = str(args.bootstrap_receipt_sha256).lower()
    if len(bootstrap_sha) != 64 or any(ch not in "0123456789abcdef" for ch in bootstrap_sha):
        raise ValueError("--bootstrap-receipt-sha256은 64자리 SHA-256이어야 합니다")
    stop_step = int(args.stop_step)
    final_step = int(args.final_step)
    if not 200 <= stop_step <= 500 or not stop_step < final_step <= 500:
        raise ValueError("A100 smoke는 200 <= stop-step < final-step <= 500 이어야 합니다")
    _preflight_a100(cublas_workspace_config=args.cublas_workspace_config)

    # 시작 전 resolved target을 한 번 만들고, target root에 과거/부분 산출물이 있으면
    # 덮어쓰지 않는다. 재시도는 새 exact source/input target 또는 수동 감사 뒤에만 한다.
    preview = _resolved_cfg(
        config,
        bootstrap_sha256=bootstrap_sha,
        label="uninterrupted",
        run_until_step=final_step,
    )
    target = str(preview["smoke_target_sha256"])
    target_root = smoke_run_directory(preview, repo_root=REPO_ROOT).parent
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError(
            "A100 smoke target root가 이미 있습니다 — 기존 receipt/partial artifact를 "
            f"감사한 뒤 새 target으로 실행하세요: {target_root}"
        )

    full_cfg = _run_train(
        config,
        bootstrap_sha256=bootstrap_sha,
        label="uninterrupted",
        run_until_step=final_step,
        cublas_workspace_config=args.cublas_workspace_config,
    )
    resumed_initial_cfg = _run_train(
        config,
        bootstrap_sha256=bootstrap_sha,
        label="resumed",
        run_until_step=stop_step,
        cublas_workspace_config=args.cublas_workspace_config,
    )
    full_dir = smoke_run_directory(full_cfg, repo_root=REPO_ROOT)
    resumed_dir = smoke_run_directory(resumed_initial_cfg, repo_root=REPO_ROOT)
    resume_last = resumed_dir / "ckpt" / "last.pt"
    stop_checkpoint = resumed_dir / "ckpt" / "stop.pt"
    _link_stop_checkpoint(resume_last, stop_checkpoint)
    resume_input_path = target_root / "resume_input.json"
    write_json_exclusive(
        resume_input_path,
        build_a100_pretrain_smoke_resume_input(
            repo_root=REPO_ROOT,
            smoke_target_sha256=target,
            resume_checkpoint=stop_checkpoint,
            stop_checkpoint=stop_checkpoint,
        ),
    )

    resumed_cfg = _run_train(
        config,
        bootstrap_sha256=bootstrap_sha,
        label="resumed",
        run_until_step=final_step,
        cublas_workspace_config=args.cublas_workspace_config,
        resume=stop_checkpoint,
    )
    if str(resumed_cfg["smoke_target_sha256"]) != target:
        raise RuntimeError("resumed smoke target이 uninterrupted target과 다릅니다")

    environment_receipt_path = target_root / "environment_receipt.json"
    write_json_exclusive(
        environment_receipt_path,
        build_a100_pretrain_smoke_environment_receipt(
            repo_root=REPO_ROOT,
            smoke_target_sha256=target,
            uninterrupted_environment=full_dir / "environment.json",
            resumed_environment=resumed_dir / "environment.json",
        ),
    )

    combined_telemetry, receipt = build_a100_pretrain_smoke_artifacts(
        repo_root=REPO_ROOT,
        smoke_target_sha256=target,
        uninterrupted_checkpoint=full_dir / "ckpt" / "last.pt",
        stop_checkpoint=stop_checkpoint,
        resumed_checkpoint=resumed_dir / "ckpt" / "last.pt",
        uninterrupted_telemetry=_phase_path(full_dir, 0, final_step),
        stopped_telemetry=_phase_path(resumed_dir, 0, stop_step),
        resumed_telemetry=_phase_path(resumed_dir, stop_step, final_step),
        environment_receipt=environment_receipt_path,
        resume_input=resume_input_path,
    )
    combined_path = target_root / "telemetry.json"
    write_json_exclusive(combined_path, combined_telemetry)
    final_receipt = finalize_a100_pretrain_smoke_receipt(
        receipt, telemetry_reference=_sha_ref(combined_path)
    )
    receipt_path = target_root / "receipt.json"
    write_json_exclusive(receipt_path, final_receipt)
    validate_a100_pretrain_smoke_receipt(
        final_receipt,
        repo_root=REPO_ROOT,
        expected_smoke_target_sha256=target,
    )

    ledger_fragment = {
        "a100_smoke_resume": {
            "evidence": _sha_ref(receipt_path),
            "environment_receipt": _sha_ref(environment_receipt_path),
            "telemetry": _sha_ref(combined_path),
        }
    }
    print("[a100 smoke] PASS — campaign ledger에 아래 fragment를 정확히 사용하세요.")
    print(json.dumps(ledger_fragment, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
