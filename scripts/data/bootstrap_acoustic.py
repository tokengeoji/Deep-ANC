#!/usr/bin/env python3
"""CPU Docker에서 네트워크/음원 다운로드 없이 acoustic 준비 receipt·합성 RIR 생성."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, validate_duct
from deep_anc.data.drive_inventory import analyze_drive_inventory


def safe_path(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("상위 이동/심볼릭 링크 경로는 허용하지 않습니다")
    return path


def build_preparation(out: Path, duct_path: Path, inventory_path: Path | None, *, n_rirs=300) -> dict:
    if os.environ.get("DEEP_ANC_CONTAINER") not in {"cpu", "jetson"}:
        raise ValueError("scripts/docker/dev.sh exec로 컨테이너 안에서 실행하세요")
    if type(n_rirs) is not int or n_rirs < 20:
        raise ValueError("RIR train/val/test 분할을 위해 20개 이상이 필요합니다")
    out, duct_path = safe_path(out), safe_path(duct_path)
    if out.exists():
        raise FileExistsError("기존 결과 폴더는 덮어쓰지 않습니다")
    inventory = None
    inventory_hash = None
    if inventory_path is not None:
        raw = safe_path(inventory_path).read_bytes()
        inventory_hash = hashlib.sha256(raw).hexdigest()
        inventory = analyze_drive_inventory(json.loads(raw))
    import numpy as np
    import yaml
    from deep_anc.dsp.duct_sim import build_rir_bank
    duct_bytes = duct_path.read_bytes()
    duct = yaml.safe_load(duct_bytes)
    validate_duct(duct)
    # 의존성 설치/변경·CUDA 검사는 하지 않는다.
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError("패키지 정합성 검사 실패: " + check.stdout + check.stderr)
    bank = build_rir_bank(duct, 48000, n_variants=n_rirs, seed=20260915, ir_len=8192)
    if any(a.shape != (n_rirs, 8192) or not np.isfinite(a).all() for a in bank.values()):
        raise ValueError("합성 RIR 배열 검증 실패")
    out.mkdir(parents=True)
    with (out / "duct_rirs_v1.npz").open("xb") as stream:
        np.savez_compressed(stream, **bank, sample_rate=48000, seed=20260915)
    rir_sha = hashlib.sha256((out / "duct_rirs_v1.npz").read_bytes()).hexdigest()
    report = {
        "schema_version": 1, "pc_bootstrap_complete": True,
        "dataset_training_ready": False, "jetson_ready": False,
        "physical_performance_claim_allowed": False,
        "storage_policy": "drive_archive_with_temporary_local_staging",
        "operation_scope": "bootstrap_without_raw_download",
        "research_use": "noncommercial_academic",
        "raw_downloaded": False, "gpu_training_started": False,
        "package_check": check.stdout.strip(), "reference_mode": "acoustic",
        "digital_reference_lead_samples": 0,
        "synthetic_rir": {"path": "duct_rirs_v1.npz", "sha256": rir_sha,
                          "duct_config_sha256": hashlib.sha256(duct_bytes).hexdigest(),
                          "sample_rate": 48000, "count": n_rirs, "ir_length": 8192,
                          "seed": 20260915, "measured": False},
        "drive_inventory_sha256": inventory_hash, "drive_diagnostic": inventory,
        "remaining": ["학습 환경에서 명시적으로 제공한 음원·manifest 전수 QA",
                      "새 acoustic 설정의 독립 학습/평가", "Jetson I/O 및 덕트 실측"],
    }
    (out / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Acoustic PC 준비 receipt\n\n"
        "네트워크/원본 음원 다운로드 없이 패키지 검증과 합성 RIR 생성을 마쳤습니다.\n"
        "학습 데이터 READY나 Jetson 실기 PASS가 아닙니다. 원본은 Drive에 보관합니다.\n"
        "RIR은 기하 시뮬레이션이며 실측 경로를 대체하지 않습니다.\n"
        "학습 환경의 명시적 데이터 QA 후 data.rir_bank에 이 NPZ 경로를 지정하세요.\n"
    )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duct", type=Path, default=REPO_ROOT / "configs/duct.yaml")
    p.add_argument("--drive-inventory", type=Path)
    p.add_argument("--n-rirs", type=int, default=300)
    args = p.parse_args()
    try:
        report = build_preparation(args.out, args.duct, args.drive_inventory, n_rirs=args.n_rirs)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[실패] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"pc_bootstrap_complete": report["pc_bootstrap_complete"],
                      "dataset_training_ready": False, "output": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
