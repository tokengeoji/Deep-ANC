#!/usr/bin/env python3
"""Elice bootstrap 산출물에서 Stage-2 공개 사전학습 artifact를 no-replace 발행한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.source_trust import (  # noqa: E402
    SourceTrustError,
    exact_clean_source_evidence,
)
from deep_anc.data.stage2_pretrain_data_issuer import (  # noqa: E402
    Stage2PretrainDataIssueError,
    build_stage2_pretrain_data_payloads,
    publish_payloads_noreplace,
    seal_published_payloads_noreplace,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (  # noqa: E402
    validate_stage2_pretrain_data_candidate,
)


def _inside(raw: str, *, label: str, must_exist: bool) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise Stage2PretrainDataIssueError(f"{label}는 repository 내부 상대경로여야 합니다")
    root = REPO_ROOT.resolve(strict=True)
    target = root / candidate
    try:
        resolved = target.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise Stage2PretrainDataIssueError(f"{label} 경로가 유효하지 않습니다") from exc
    cursor = root
    for part in candidate.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise Stage2PretrainDataIssueError(f"{label} 경로에 symlink가 있습니다")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--manifest-dir", default="data/manifests/canonical_v4"
    )
    parser.add_argument("--plant-binding", required=True)
    parser.add_argument("--expected-plant-binding-sha256", required=True)
    parser.add_argument(
        "--bootstrap-receipt",
        default="data/manifests/elice_bootstrap_receipt.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="기본: data/manifests/stage2_2khz/<expected commit>",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, int(os.cpu_count() or 1))),
        help="actual source SHA/decode/Welch worker 수(1..16)",
    )
    args = parser.parse_args(argv)
    try:
        if isinstance(args.workers, bool) or not 1 <= int(args.workers) <= 16:
            raise Stage2PretrainDataIssueError("--workers는 1..16이어야 합니다")
        expected_commit = str(args.expected_commit).lower()
        # import가 끝난 뒤에도 tracked/untracked protected source를 byte 단위로 다시
        # 검사한다. data/ 아래 bootstrap/raw는 repository ignore 정책에 따라 허용된다.
        exact_clean_source_evidence(
            REPO_ROOT,
            expected_commit=expected_commit,
            reject_runtime_bytecode=False,
        )
        manifest_dir = _inside(args.manifest_dir, label="manifest dir", must_exist=True)
        if not manifest_dir.is_dir() or manifest_dir.is_symlink():
            raise Stage2PretrainDataIssueError("manifest dir는 non-symlink directory여야 합니다")
        plant_binding = _inside(
            args.plant_binding, label="plant binding", must_exist=True
        )
        bootstrap = _inside(
            args.bootstrap_receipt, label="bootstrap receipt", must_exist=True
        )
        output_raw = (
            f"data/manifests/stage2_2khz/{expected_commit}"
            if args.output_dir is None
            else args.output_dir
        )
        output = _inside(output_raw, label="output dir", must_exist=False)
        required_parent = (REPO_ROOT / "data/manifests/stage2_2khz").resolve(
            strict=False
        )
        try:
            output.relative_to(required_parent)
        except ValueError as exc:
            raise Stage2PretrainDataIssueError(
                "output dir는 data/manifests/stage2_2khz 아래여야 합니다"
            ) from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        payloads = build_stage2_pretrain_data_payloads(
            REPO_ROOT.resolve(strict=True),
            manifest_dir=manifest_dir,
            plant_binding_path=plant_binding,
            expected_plant_binding_sha256=str(
                args.expected_plant_binding_sha256
            ).lower(),
            source_inventory_commit_sha=expected_commit,
            bootstrap_receipt_path=bootstrap.relative_to(REPO_ROOT).as_posix(),
            workers=int(args.workers),
        )
        digests = publish_payloads_noreplace(output, payloads)
        refs = {
            name: (
                (output / filename).relative_to(REPO_ROOT).as_posix(),
                digests[filename],
            )
            for name, filename in {
                "manifest": "manifest_bundle.json",
                "lineage": "lineage_receipt.json",
                "coverage": "frequency_coverage_receipt.json",
                "bootstrap": "transfer_bootstrap_receipt.json",
            }.items()
        }
        binding = validate_stage2_pretrain_data_candidate(
            repository_root=REPO_ROOT,
            manifest_ref=refs["manifest"],
            lineage_ref=refs["lineage"],
            coverage_ref=refs["coverage"],
            bootstrap_ref=refs["bootstrap"],
            plant_binding_file_sha256=str(
                args.expected_plant_binding_sha256
            ).lower(),
            workers=int(args.workers),
        )
        completion, completion_sha = seal_published_payloads_noreplace(
            output,
            artifact_sha256=digests,
            validated_record_count=len(binding.records),
        )
        result = {
            "schema": "stage2_2khz_public_data_issue_cli_v1",
            "status": "PASS_CANDIDATE_REQUIRES_GIT_AUTHORITY",
            "source_inventory_commit_sha": expected_commit,
            "output_dir": output.relative_to(REPO_ROOT).as_posix(),
            "artifact_sha256": dict(sorted(digests.items())),
            "publication_complete": {
                "path": (
                    (output / "publication_complete.json")
                    .relative_to(REPO_ROOT)
                    .as_posix()
                ),
                "sha256": completion_sha,
                "status": completion["status"],
            },
            "validated_record_count": len(binding.records),
            "audio_opened": False,
            "gpu_initialized": False,
            "network_used": False,
            "issuer_workers": int(args.workers),
            "validator_workers": int(args.workers),
        }
    except (
        OSError,
        RuntimeError,
        SourceTrustError,
        Stage2PretrainDataIssueError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"[BLOCKED] Stage-2 public data issuer: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
