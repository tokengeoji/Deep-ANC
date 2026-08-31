#!/usr/bin/env python3
"""Stage-2 scratch pretrain sampler/DNH/criterion/external artifact 발행 CLI.

``issue``는 CPU에서만 actual manifest source bytes와 measured P/S를 소비한다. 실제
family-balanced batch를 fresh scratch Tiny에 통과시킨 뒤 raw tensor NPZ를 먼저
no-replace 발행하고 reload해 DNH gradient share를 계산한다. ``external``은 이
artifact들을 profile/authority와 함께 커밋한 다음 별도 commit candidate를 발행한다.

오디오 장치를 열거나 네트워크/GPU를 사용하는 코드는 이 파일에 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from deep_anc.models import build_model  # noqa: E402
from deep_anc.train.stage2_2khz_binding import (  # noqa: E402
    load_stage2_2khz_plant_binding,
)
from deep_anc.train.stage2_2khz_execution import (  # noqa: E402
    Stage2ActualBatchIdentity,
    Stage2CausalPrefixAdapter,
    Stage2FamilyComponentBatchSampler,
    require_stage2_actuator_limit,
)
from deep_anc.train.stage2_2khz_git_authority import (  # noqa: E402
    verify_tracked_head_file,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (  # noqa: E402
    load_stage2_pretrain_data_binding,
    load_stage2_pretrain_typed_admission,
)
from deep_anc.train.stage2_2khz_pretrain_issuer import (  # noqa: E402
    STAGE2_PRETRAIN_ISSUER_SUMMARY_SCHEMA,
    build_canonical_pretrain_external_contract,
    build_criterion_receipt,
    build_dnh_receipt,
    build_sampler_receipt,
    calibrate_dnh_from_reloaded_batch,
    load_calibration_batch,
    model_initial_state_sha256,
    publish_calibration_batch_no_replace,
    publish_json_no_replace,
    snapshot_actual_stage2_batch,
)
from deep_anc.train.stage2_2khz_runner import Stage2PublicTensorLoader  # noqa: E402


SCRIPT_RELATIVE_PATH = "scripts/train/issue_stage2_2khz_pretrain_contract.py"
DEFAULT_DUCT_PROFILE = "configs/stage2_2khz_duct.yaml"
DEFAULT_DATA_PROFILE = "configs/stage2_2khz_data.yaml"
DEFAULT_PRETRAIN_PROFILE = "configs/stage2_2khz_train_pretrain.yaml"
DEFAULT_CAMPAIGN_PROFILE = "configs/stage2_2khz_campaign.yaml"
# actual tensor NPZ는 추적하지 않는다. 기본 경로가 Git working tree를 더럽히면
# 다음 exact-clean authority 검증 자체가 실패하므로 root-held ignored results를 쓴다.
DEFAULT_OUTPUT_ROOT = "results/stage2_2khz_pretrain_contracts"

_GENERATION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _snapshot_runtime_ref(root: Path, reference: tuple[str, str], *, label: str) -> str:
    """gitignored 대용량 artifact를 nofollow snapshot하고 declared SHA와 대조한다."""

    path = _inside(root, reference[0], label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}는 regular file이어야 합니다")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"{label}가 snapshot 중 바뀌었습니다")
    if size != int(after.st_size) or digest.hexdigest() != reference[1]:
        raise ValueError(f"{label} runtime bytes SHA가 declared ref와 다릅니다")
    return digest.hexdigest()


def _inside(root: Path, value: str, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} 경로에 symlink가 있습니다")
    return root / relative


def _yaml_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label}는 UTF-8 YAML이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root는 mapping이어야 합니다")
    return payload


def _tracked_yaml(root: Path, relative: str, *, production: bool) -> tuple[dict[str, Any], str, str | None]:
    if production:
        content, digest, head = verify_tracked_head_file(root, relative)
        return _yaml_bytes(content, label=relative), digest, head
    path = _inside(root, relative, label=relative)
    content = path.read_bytes()
    return _yaml_bytes(content, label=relative), _sha256(content), None


def _ref(section: Mapping[str, Any], key: str, *, label: str) -> tuple[str, str]:
    entry = section.get(key)
    if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
        raise ValueError(f"{label}.{key} ref key 집합이 exact하지 않습니다")
    path = entry.get("path")
    sha = entry.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"{label}.{key}가 아직 null이거나 SHA가 잘못됐습니다")
    return path, sha


def _print_plan(args: argparse.Namespace, *, blockers: list[str]) -> None:
    payload = {
        "mode": args.command,
        "dry_run": True,
        "gpu_used": False,
        "audio_used": False,
        "network_used": False,
        "generation": getattr(args, "generation", None),
        "output_root": getattr(args, "output_root", None),
        "production_steps": [
            "origin/dev exact clean tracked CLI/profile preflight",
            "production P/S authority 및 actual public source bytes 재검증",
            "family→component→item global-index deterministic batch 생성",
            "fresh scratch Tiny CPU causal P*n/S*y forward",
            "actual tensor NPZ no-replace 저장 후 reload/hash 검증",
            "output-y DNH 0.2/0.3/0.4 후보 full-objective autograd 재계산",
            "sampler/DNH/criterion JSON no-replace 발행",
            "profile/authority commit 뒤 external subcommand로 candidate 발행",
        ],
        "blockers": blockers,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _dry_run_issue(args: argparse.Namespace, root: Path) -> int:
    blockers: list[str] = []
    try:
        duct, _, _ = _tracked_yaml(root, args.duct_profile, production=False)
        for key in ("primary_path", "secondary_path", "plant_binding"):
            _ref(duct.get("artifacts", {}), key, label="duct.artifacts")
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
    try:
        data, _, _ = _tracked_yaml(root, args.data_profile, production=False)
        for key in (
            "manifest_bundle",
            "lineage_receipt",
            "frequency_coverage_receipt",
            "transfer_bootstrap_receipt",
        ):
            _ref(data.get("artifacts", {}), key, label="data.artifacts")
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
    try:
        train, _, _ = _tracked_yaml(root, args.pretrain_profile, production=False)
        model_ref = train["execution"]["model_config"]
        _ref({"model": model_ref}, "model", label="pretrain.execution")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append(f"pretrain profile: {exc}")
    _print_plan(args, blockers=blockers)
    return 0


def _seed_cpu(seed: int, *, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(threads))
    if torch.cuda.is_initialized():
        raise RuntimeError("Stage-2 issuer 실행 전에 CUDA context가 이미 초기화됐습니다")


def _issue(args: argparse.Namespace, root: Path) -> int:
    if args.dry_run:
        return _dry_run_issue(args, root)
    if not _GENERATION.fullmatch(args.generation):
        raise ValueError("generation은 1..96자의 소문자/숫자/._-만 허용합니다")
    # corpus/P/S scan 또는 output directory 생성보다 먼저 exact checkout을 닫는다.
    _, _, execution_head = verify_tracked_head_file(root, SCRIPT_RELATIVE_PATH)
    duct, _, duct_head = _tracked_yaml(root, args.duct_profile, production=True)
    data, _, data_head = _tracked_yaml(root, args.data_profile, production=True)
    train, _, train_head = _tracked_yaml(root, args.pretrain_profile, production=True)
    if {execution_head, duct_head, data_head, train_head} != {execution_head}:
        raise ValueError("Stage-2 issuer/profile execution HEAD가 다릅니다")

    duct_artifacts = duct.get("artifacts", {})
    primary_ref = _ref(duct_artifacts, "primary_path", label="duct.artifacts")
    secondary_ref = _ref(duct_artifacts, "secondary_path", label="duct.artifacts")
    binding_ref = _ref(duct_artifacts, "plant_binding", label="duct.artifacts")
    data_artifacts = data.get("artifacts", {})
    manifest_ref = _ref(data_artifacts, "manifest_bundle", label="data.artifacts")
    lineage_ref = _ref(data_artifacts, "lineage_receipt", label="data.artifacts")
    coverage_ref = _ref(
        data_artifacts, "frequency_coverage_receipt", label="data.artifacts"
    )
    bootstrap_ref = _ref(
        data_artifacts, "transfer_bootstrap_receipt", label="data.artifacts"
    )
    execution = train.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Stage-2 pretrain execution profile이 없습니다")
    model_path, model_sha = _ref(
        {"model": execution.get("model_config")},
        "model",
        label="pretrain.execution",
    )
    model_bytes, actual_model_sha, model_head = verify_tracked_head_file(root, model_path)
    if model_head != execution_head or actual_model_sha != model_sha:
        raise ValueError("Stage-2 model config tracked bytes SHA가 profile과 다릅니다")
    model_cfg = _yaml_bytes(model_bytes, label="Stage-2 model config")
    if int(model_cfg.get("in_channels", 0)) != 2:
        raise ValueError("Stage-2 issuer는 2-channel digital-reference Tiny만 허용합니다")
    actuator_limit = require_stage2_actuator_limit(model_cfg)
    implementation_paths = (
        "src/deep_anc/losses/stage2_2khz_loss.py",
        "src/deep_anc/train/stage2_2khz_execution.py",
        "src/deep_anc/train/stage2_2khz_runner.py",
    )
    implementation_sha: dict[str, str] = {}
    for relative in implementation_paths:
        _, digest, head = verify_tracked_head_file(root, relative)
        if head != execution_head:
            raise ValueError("Stage-2 implementation/execution HEAD가 다릅니다")
        implementation_sha[relative] = digest

    binding = load_stage2_2khz_plant_binding(
        binding_ref[0],
        repository_root=root,
        expected_binding_file_sha256=binding_ref[1],
    )
    if binding.fixture_only:
        raise ValueError("Stage-2 production issuer는 fixture plant binding을 거부합니다")
    if (
        binding.primary_path_sha256 != primary_ref[1]
        or binding.secondary_path_sha256 != secondary_ref[1]
    ):
        raise ValueError("Stage-2 duct profile P/S와 typed binding이 다릅니다")
    data_binding = load_stage2_pretrain_data_binding(
        repository_root=root,
        manifest_ref=manifest_ref,
        lineage_ref=lineage_ref,
        coverage_ref=coverage_ref,
        bootstrap_ref=bootstrap_ref,
        plant_binding_file_sha256=binding.binding_file_sha256,
    )

    seed = int(train.get("seed", -1))
    batch_size = int(execution.get("batch_size", -1))
    target_samples = int(execution.get("target_samples", -1))
    if seed != 20260803 or batch_size != 96 or target_samples != 16_384:
        raise ValueError("Stage-2 canonical pretrain seed/batch/target가 exact하지 않습니다")
    output_root = _inside(root, args.output_root, label="output root")
    output_root.mkdir(parents=True, exist_ok=True)
    generation_dir = output_root / args.generation
    generation_dir.mkdir(mode=0o755, exist_ok=False)

    sampler_path = generation_dir / "sampler_receipt.json"
    provisional = Stage2FamilyComponentBatchSampler(
        data_binding.records,
        batch_size=batch_size,
        seed=seed,
        split="train",
        manifest_bundle_sha256=data_binding.manifest_bundle_sha256,
        sampler_receipt_sha256="0" * 64,
    )
    sampler_sha = publish_json_no_replace(
        sampler_path, build_sampler_receipt(provisional)
    )
    sampler = Stage2FamilyComponentBatchSampler(
        data_binding.records,
        batch_size=batch_size,
        seed=seed,
        split="train",
        manifest_bundle_sha256=data_binding.manifest_bundle_sha256,
        sampler_receipt_sha256=sampler_sha,
    )
    identity = Stage2ActualBatchIdentity.from_sampler(
        sampler, global_step=int(args.global_step), rank=0, world_size=1
    )

    _seed_cpu(seed, threads=int(args.cpu_threads))
    model = build_model(model_cfg).cpu().eval()
    initial_state_sha = model_initial_state_sha256(model)
    admission_view = SimpleNamespace(
        plant_binding=binding,
        data_binding=data_binding,
    )
    pipeline = execution.get("data_pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("Stage-2 bounded data pipeline config가 없습니다")
    loader = Stage2PublicTensorLoader(
        repository_root=root,
        admission=admission_view,  # production typed P/S+data attrs만 소비
        target_samples=target_samples,
        cache_items=min(int(pipeline["source_cache_items"]), batch_size),
        valid_start_candidates=int(pipeline["valid_start_candidates_per_source"]),
        pin_memory=False,
        model_actuator_limit_abs=actuator_limit,
    )
    tensor_batch = loader.build(identity)
    adapter = Stage2CausalPrefixAdapter.from_verified_binding(binding).cpu()
    with torch.no_grad():
        causal_result = adapter(model, tensor_batch.causal)
    snapshot = snapshot_actual_stage2_batch(
        tensor_batch=tensor_batch,
        identity=identity,
        causal_result=causal_result,
        binding=binding,
        model_config_sha256=model_sha,
        model_initial_state_sha256_value=initial_state_sha,
    )
    batch_path = generation_dir / "actual_calibration_batch.npz"
    batch_sha = publish_calibration_batch_no_replace(batch_path, snapshot)
    reloaded = load_calibration_batch(batch_path, expected_sha256=batch_sha)
    calibration = calibrate_dnh_from_reloaded_batch(reloaded, binding=binding)

    dnh_path = generation_dir / "dnh_calibration_receipt.json"
    dnh_sha = publish_json_no_replace(
        dnh_path,
        build_dnh_receipt(
            snapshot=reloaded,
            calibration_batch_path=batch_path.relative_to(root).as_posix(),
            calibration_batch_sha256=batch_sha,
            calibration=calibration,
        ),
    )
    sampler_relative = sampler_path.relative_to(root).as_posix()
    dnh_relative = dnh_path.relative_to(root).as_posix()
    criterion_path = generation_dir / "criterion_implementation_receipt.json"
    criterion_payload = build_criterion_receipt(
        repository_root=root,
        plant_binding_file_sha256=binding.binding_file_sha256,
        manifest_bundle_sha256=data_binding.manifest_bundle_sha256,
        sampler_receipt_path=sampler_relative,
        sampler_receipt_sha256=sampler_sha,
        dnh_receipt_path=dnh_relative,
        dnh_receipt_sha256=dnh_sha,
        model_config_path=model_path,
        model_config_sha256=model_sha,
        model_initial_state_sha256_value=initial_state_sha,
        batch_size=batch_size,
        seed=seed,
    )
    for key in (
        "loss_implementation",
        "trainer_adapter_implementation",
        "scratch_runner_implementation",
    ):
        entry = criterion_payload[key]
        if implementation_sha[str(entry["path"])] != str(entry["sha256"]):
            raise RuntimeError(
                "Stage-2 calibration 중 implementation bytes가 바뀌었습니다; "
                "현재 generation은 incomplete로 보존하고 새 generation을 사용하세요"
            )
    criterion_sha = publish_json_no_replace(criterion_path, criterion_payload)
    summary_path = generation_dir / "issuer_summary.json"
    summary = {
        "schema": STAGE2_PRETRAIN_ISSUER_SUMMARY_SCHEMA,
        "status": "CANDIDATE_ARTIFACTS_COMPLETE_REQUIRES_HUMAN_GIT_AUTHORITY",
        "canonical_pretrain_eligible": False,
        "reason": "criterion/profile/authority/external contract는 별도 review+commit 필요",
        "execution_commit_sha": execution_head,
        "gpu_used": False,
        "audio_used": False,
        "network_used": False,
        "loader_worker_count": 0,
        "global_step": int(args.global_step),
        "actual_source_bytes_reverified": True,
        "actual_measured_primary_secondary_used": True,
        "actual_family_balanced_batch_used": True,
        "calibration_reloaded_before_receipt": True,
        "gradient_calibration": calibration,
        "artifacts": {
            "sampler_receipt": {
                "path": sampler_relative,
                "sha256": sampler_sha,
            },
            "actual_calibration_batch": {
                "path": batch_path.relative_to(root).as_posix(),
                "sha256": batch_sha,
            },
            "dnh_calibration_receipt": {
                "path": dnh_relative,
                "sha256": dnh_sha,
            },
            "criterion_implementation_receipt": {
                "path": criterion_path.relative_to(root).as_posix(),
                "sha256": criterion_sha,
            },
        },
        "next_required_action": (
            "artifact/profile/authority를 review+commit+push한 뒤 external subcommand 실행"
        ),
    }
    summary_sha = publish_json_no_replace(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "CANDIDATE_ARTIFACTS_COMPLETE",
                "generation": args.generation,
                "criterion_receipt": {
                    "path": criterion_path.relative_to(root).as_posix(),
                    "sha256": criterion_sha,
                },
                "summary": {
                    "path": summary_path.relative_to(root).as_posix(),
                    "sha256": summary_sha,
                },
                "training_ready": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _external(args: argparse.Namespace, root: Path) -> int:
    if args.dry_run:
        _print_plan(args, blockers=[])
        return 0
    _, _, execution_head = verify_tracked_head_file(root, SCRIPT_RELATIVE_PATH)
    campaign, _, campaign_head = _tracked_yaml(
        root, args.campaign_profile, production=True
    )
    if campaign_head != execution_head:
        raise ValueError("Stage-2 campaign/issuer HEAD가 다릅니다")
    profiles = campaign.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("Stage-2 campaign profiles가 없습니다")
    profile_sha: dict[str, str] = {}
    profile_payload: dict[str, dict[str, Any]] = {}
    for role in (
        "duct",
        "data",
        "evaluation",
        "canonical_pretrain",
        "canonical_finetune",
    ):
        path, declared_sha = _ref(profiles, role, label="campaign.profiles")
        payload, actual_sha, head = _tracked_yaml(root, path, production=True)
        if head != execution_head or actual_sha != declared_sha:
            raise ValueError(f"Stage-2 campaign {role} tracked SHA가 다릅니다")
        profile_sha[role] = actual_sha
        profile_payload[role] = payload
    duct = profile_payload["duct"]["artifacts"]
    data = profile_payload["data"]["artifacts"]
    train = profile_payload["canonical_pretrain"]
    primary = _ref(duct, "primary_path", label="duct.artifacts")
    secondary = _ref(duct, "secondary_path", label="duct.artifacts")
    binding = _ref(duct, "plant_binding", label="duct.artifacts")
    manifest = _ref(data, "manifest_bundle", label="data.artifacts")
    criterion = _ref(
        {"criterion": train["criterion"]["implementation_receipt"]},
        "criterion",
        label="pretrain.criterion",
    )
    # 대용량 P/S/manifest/receipt는 의도적으로 gitignored일 수 있다. tracked file을
    # 요구하지 않고 nofollow held bytes + typed Git authority chain으로 검증한다.
    for reference, label in (
        (primary, "external primary P"),
        (secondary, "external secondary S"),
        (binding, "external plant binding"),
        (manifest, "external manifest"),
        (criterion, "external criterion receipt"),
    ):
        _snapshot_runtime_ref(root, reference, label=label)
    data_artifacts = profile_payload["data"]["artifacts"]
    typed = load_stage2_pretrain_typed_admission(
        repository_root=root,
        primary_path_sha256=primary[1],
        secondary_path_sha256=secondary[1],
        plant_binding_ref=binding,
        manifest_ref=manifest,
        lineage_ref=_ref(
            data_artifacts, "lineage_receipt", label="data.artifacts"
        ),
        coverage_ref=_ref(
            data_artifacts, "frequency_coverage_receipt", label="data.artifacts"
        ),
        bootstrap_ref=_ref(
            data_artifacts, "transfer_bootstrap_receipt", label="data.artifacts"
        ),
        criterion_receipt_ref=criterion,
    )
    if typed.status != "READY":  # pragma: no cover - typed contract guard
        raise ValueError("Stage-2 typed pretrain admission이 READY가 아닙니다")
    payload = build_canonical_pretrain_external_contract(
        artifact_source_commit_sha=execution_head,
        profile_sha256=profile_sha,
        plant_sha256={
            "primary_path_sha256": primary[1],
            "secondary_path_sha256": secondary[1],
            "plant_binding_sha256": binding[1],
        },
        manifest_bundle_sha256=manifest[1],
        criterion_receipt_sha256=criterion[1],
    )
    output = _inside(root, args.output, label="external output")
    digest = publish_json_no_replace(output, payload)
    print(
        json.dumps(
            {
                "status": "EXTERNAL_CANDIDATE_COMPLETE_REQUIRES_COMMIT",
                "path": output.relative_to(root).as_posix(),
                "sha256": digest,
                "training_ready": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    issue = subparsers.add_parser("issue", help="actual batch/DNH/criterion candidate 발행")
    issue.add_argument("--duct-profile", default=DEFAULT_DUCT_PROFILE)
    issue.add_argument("--data-profile", default=DEFAULT_DATA_PROFILE)
    issue.add_argument("--pretrain-profile", default=DEFAULT_PRETRAIN_PROFILE)
    issue.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    issue.add_argument("--generation", required=True)
    issue.add_argument("--global-step", type=int, default=0)
    issue.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    issue.add_argument("--dry-run", action="store_true")

    external = subparsers.add_parser(
        "external", help="committed profile chain의 external candidate 발행"
    )
    external.add_argument("--campaign-profile", default=DEFAULT_CAMPAIGN_PROFILE)
    external.add_argument("--output", required=True)
    external.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).resolve(strict=True)
    if args.command == "issue":
        if args.global_step < 0:
            raise ValueError("global-step은 0 이상이어야 합니다")
        if not 1 <= args.cpu_threads <= 64:
            raise ValueError("cpu-threads는 1..64여야 합니다")
        return _issue(args, root)
    if args.command == "external":
        return _external(args, root)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
