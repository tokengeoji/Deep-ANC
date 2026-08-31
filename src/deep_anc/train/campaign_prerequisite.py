"""canonical pretrain을 열기 전 campaign 증거 ledger 검증."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .evaluation_contract import FileSnapshot, snapshot_regular_file
from .campaign_evidence import (
    MEASURED_PROBE_SELECTION_SCORE,
    PILOT_SELECTION_RULE,
    select_loss_pilot,
    validate_canonical_evidence_target,
    validate_g0_receipt,
    validate_gradient_budget_receipt,
    validate_loss_pilot_candidate,
    validate_measured_probe,
    validate_prepilot_gradient_receipt,
)
from .a100_pretrain_smoke import (
    SMOKE_ROOT,
    build_a100_pretrain_smoke_target,
    validate_a100_pretrain_smoke_receipt,
)


SCHEMA_VERSION = 7
CANONICAL_PATH = "results/training_prerequisites/canonical_pretrain.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}가 64자리 SHA-256이 아닙니다")
    return text


def _path_inside(root: Path, value: object, *, label: str) -> Path:
    path = Path(str(value)).expanduser()
    target = Path(os.path.abspath(path if path.is_absolute() else root / path))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    if root.is_symlink():
        raise ValueError(f"저장소 root는 심볼릭 링크일 수 없습니다: {root}")
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(
                f"{label} 경로에 심볼릭 링크가 있어 저장소 밖을 가리킬 수 있습니다: {cursor}"
            )
    try:
        target.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}의 resolved path가 저장소 밖입니다: {target}") from exc
    return target


def _evidence_snapshot(root: Path, item: object, *, label: str) -> FileSnapshot:
    entry = _exact_keys(item, {"path", "sha256"}, label=label)
    snapshot = snapshot_regular_file(_path_inside(root, entry["path"], label=label))
    if snapshot.sha256 != _sha(entry["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 ledger와 다릅니다")
    return snapshot


def validate_canonical_pretrain_ledger_payload(
    cfg: dict,
    ledger_payload: object,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """이미 만든 ledger payload의 raw evidence를 파일 공개 전에도 검증한다.

    이 함수는 external ledger SHA/path trust anchor를 일부러 읽지 않는다. issuer는
    prospective JSON digest로 resolve한 canonical cfg와 이 함수를 먼저 통과시킨 뒤
    no-replace 공개하고, :func:`validate_canonical_pretrain_prerequisites`가 공개된
    bytes/anchor를 한 번 더 닫는다. 따라서 실패한 raw G0/pilot/probe가 canonical
    pathname에 남는 일이 없다.
    """

    if str(cfg.get("experiment_role", "")) != "canonical_pretrain":
        return {}
    root = Path(os.path.abspath(Path(repo_root)))
    if cfg.get("seed") == 20260903:
        # secondary는 primary v7 ledger를 자기 seed cfg로 직접 검증하지 않는다.
        # 별도 no-replace prerequisite가 primary raw chain과 fresh seed-specific
        # smoke를 먼저 검증한 뒤, 그 안에서 실제 primary checkpoint cfg로 v7
        # validator를 재사용한다.
        from .second_seed_prerequisite import validate_second_seed_prerequisites

        return validate_second_seed_prerequisites(cfg, repo_root=root)
    ledger = _exact_keys(
        ledger_payload,
        {
            "schema_version",
            "source",
            "loss_pilot_selection",
            "gradient_budget",
            "a100_smoke_resume",
        },
        label="campaign prerequisite",
    )
    if ledger["schema_version"] != SCHEMA_VERSION:
        raise ValueError("campaign prerequisite schema_version이 다릅니다")

    # embedded SHA만 맞고 strict P/S bytes가 나중에 바뀐 cfg는 canonical을 열 수
    # 없다. target validator는 둘을 같은 순간에 다시 snapshot한다.
    contract = validate_canonical_evidence_target(cfg, repo_root=root)
    source = _exact_keys(
        ledger["source"],
        {
            "git_commit",
            "source_tree_sha256",
            "bootstrap_receipt_sha256",
            "archive_cache_manifest_sha256",
            "primary_path_sha256",
            "secondary_path_sha256",
        },
        label="campaign source",
    )
    contract_source = contract.get("source") or {}
    artifacts = contract.get("artifacts") or {}
    input_generation = contract.get("input_generation") or {}
    expected_generation = {
        "bootstrap_receipt_sha256": (cfg.get("data") or {}).get(
            "bootstrap_receipt_sha256"
        ),
        "archive_cache_manifest_sha256": (cfg.get("data") or {}).get(
            "archive_cache_manifest_sha256"
        ),
        "transfer_manifest_sha256": (cfg.get("data") or {}).get(
            "transfer_manifest_sha256"
        ),
        "recorded_transfer_aggregate_sha256": (cfg.get("data") or {}).get(
            "recorded_transfer_aggregate_sha256"
        ),
    }
    required_generation = {
        key: value
        for key, value in expected_generation.items()
        if key != "archive_cache_manifest_sha256"
    }
    archive_manifest_sha = expected_generation["archive_cache_manifest_sha256"]
    if (
        input_generation != expected_generation
        or not all(required_generation.values())
        or (
            archive_manifest_sha is not None
            and not _SHA256.fullmatch(str(archive_manifest_sha))
        )
    ):
        raise ValueError("campaign prerequisite contract input generation이 불완전합니다")
    expected_source = {
        "git_commit": contract_source.get("git_commit"),
        "source_tree_sha256": contract_source.get("source_tree_sha256"),
        "bootstrap_receipt_sha256": (cfg.get("data") or {}).get(
            "bootstrap_receipt_sha256"
        ),
        "archive_cache_manifest_sha256": archive_manifest_sha,
        "primary_path_sha256": (artifacts.get("primary_path") or {}).get("sha256"),
        "secondary_path_sha256": (artifacts.get("secondary_path") or {}).get("sha256"),
    }
    required_source = {
        key: value
        for key, value in expected_source.items()
        if key != "archive_cache_manifest_sha256"
    }
    if source != expected_source or not all(required_source.values()):
        raise ValueError("campaign prerequisite strict P/S/source/bootstrap identity가 다릅니다")

    # schema v7은 사람이 적은 `all_finite`, NMSE, gradient share, score, winner,
    # passed boolean을 받지 않는다. 각각 receipt/checkpoint/batch/metrics bytes에서
    # 아래 validator가 다시 계산한다. 각 alpha는 자기 λ를 가진 identity
    # (alpha,frame,lambda_dnh)이며 approved G0→pre-pilot output-gradient→20k
    # surrogate→5k measured probe를 하나의 chain으로 묶는다. 한 λ를 모든
    # alpha에 강제하지 않고, 최종 loss 선택은 probe recorded-val 점수만 사용한다.

    pilot = _exact_keys(
        ledger["loss_pilot_selection"],
        {"selection_rule", "candidates"},
        label="loss pilot selection",
    )
    if pilot["selection_rule"] != PILOT_SELECTION_RULE:
        raise ValueError("loss pilot selection rule이 승인 규칙과 다릅니다")
    if not isinstance(pilot["candidates"], list):
        raise ValueError("loss pilot candidates가 list가 아닙니다")
    candidate_rows: list[dict[str, Any]] = []
    # 각 G0 evidence directory가 같은 tensor bytes를 자체 보존할 수는 있지만 SHA는
    # 모두 같아야 한다. winner G0의 concrete artifact를 authority로 고정하고,
    # selected-20k drift receipt는 그 path+SHA 자체를 재사용해야 한다.
    g0_batch_sha256: set[str] = set()
    for index, raw in enumerate(pilot["candidates"]):
        chain = _exact_keys(
            raw,
            {"g0", "gradient_calibration", "pilot", "measured_probe"},
            label=f"loss candidate chain[{index}]",
        )
        g0 = _exact_keys(
            chain["g0"], {"receipt"}, label=f"candidate G0[{index}]"
        )
        g0_row = validate_g0_receipt(
            g0["receipt"],
            repo_root=root,
            canonical_cfg=cfg,
            canonical_contract=contract,
        )
        g0_batch = g0_row["batch"]
        g0_batch_sha256.add(str(g0_batch.sha256))
        identity = tuple(g0_row["identity"])
        gradient = _exact_keys(
            chain["gradient_calibration"],
            {"receipt"},
            label=f"candidate gradient calibration[{index}]",
        )
        gradient_row = validate_prepilot_gradient_receipt(
            gradient["receipt"],
            repo_root=root,
            canonical_cfg=cfg,
            canonical_contract=contract,
            g0_receipt_reference=g0["receipt"],
            expected_identity=identity,
        )
        pilot_row = validate_loss_pilot_candidate(
            chain["pilot"],
            repo_root=root,
            canonical_cfg=cfg,
            canonical_contract=contract,
            expected_identity=identity,
            label=f"loss pilot candidate[{index}]",
        )
        if tuple(pilot_row["identity"]) != identity:
            raise ValueError(
                f"loss pilot candidate[{index}] identity가 G0/calibration과 다릅니다"
            )
        probe_row = validate_measured_probe(
            chain["measured_probe"],
            repo_root=root,
            canonical_cfg=cfg,
            canonical_contract=contract,
            expected_identity=identity,
            expected_init_checkpoint_sha256=pilot_row["best_snapshot"].sha256,
        )
        if tuple(probe_row["identity"]) != identity:
            raise ValueError(
                f"measured probe candidate[{index}] identity가 pilot과 다릅니다"
            )
        candidate_rows.append(
            {
                "identity": identity,
                "score_db": float(probe_row["score_db"]),
                "selection_score_source": MEASURED_PROBE_SELECTION_SCORE,
                "g0": g0_row,
                "gradient_calibration": gradient_row,
                "pilot": pilot_row,
                "measured_probe": probe_row,
            }
        )
    if len(g0_batch_sha256) != 1:
        raise ValueError(
            "alpha별 G0가 같은 authoritative fixed batch SHA를 사용하지 않았습니다: "
            f"{sorted(g0_batch_sha256)}"
        )
    selection = select_loss_pilot(candidate_rows)
    winner_identity = tuple(selection["winner_identity"])
    authoritative_g0_batch = selection["winner"]["g0"]["batch"]
    cfg_identity = (
        float((cfg.get("loss") or {}).get("nmse_cvar_alpha")),
        float((cfg.get("loss") or {}).get("lambda_frame")),
        float((cfg.get("loss") or {}).get("lambda_dnh")),
    )
    if cfg_identity != winner_identity:
        raise ValueError(
            "raw measured-probe winner (alpha,frame,lambda_dnh)와 canonical "
            "pretrain loss가 다릅니다"
        )

    # pre-pilot calibration은 잘못된 λ로 수시간 pilot을 쓰는 것을 막고,
    # selected-20k 검사는 학습 뒤 출력 분포에서 share가 drift하지 않았는지 닫는다.
    # 둘 중 하나도 다른 하나를 대체하지 않는다.
    gradient = _exact_keys(
        ledger["gradient_budget"], {"receipt"}, label="winner gradient budget"
    )
    validate_gradient_budget_receipt(
        gradient["receipt"],
        repo_root=root,
        canonical_cfg=cfg,
        canonical_contract=contract,
        expected_checkpoint_sha256=selection["winner"]["pilot"][
            "best_snapshot"
        ].sha256,
        expected_identity=winner_identity,
        expected_batch_path=authoritative_g0_batch.path,
        expected_batch_sha256=authoritative_g0_batch.sha256,
    )

    smoke = _exact_keys(
        ledger["a100_smoke_resume"],
        {"evidence", "environment_receipt", "telemetry"},
        label="A100 smoke resume",
    )
    receipt_snapshot = _evidence_snapshot(
        root, smoke["evidence"], label="A100 smoke receipt evidence"
    )
    _evidence_snapshot(
        root, smoke["environment_receipt"], label="A100 environment receipt"
    )
    _evidence_snapshot(
        root, smoke["telemetry"], label="A100 smoke telemetry"
    )
    try:
        receipt = json.loads(receipt_snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("A100 smoke receipt JSON이 손상됐습니다") from exc
    if not isinstance(receipt, dict):
        raise ValueError("A100 smoke receipt 최상위가 mapping이 아닙니다")
    # ledger가 가리킨 bytes와 receipt 안의 bytes reference가 서로 같아야 한다.
    # 이후 validator는 full/resumed checkpoint, phase telemetry, CUDA environment를
    # fresh snapshot으로 다시 열어 model/optimizer/scheduler/RNG까지 직접 비교한다.
    if receipt.get("environment_receipt") != smoke["environment_receipt"]:
        raise ValueError("A100 smoke receipt environment reference가 ledger와 다릅니다")
    if receipt.get("telemetry") != smoke["telemetry"]:
        raise ValueError("A100 smoke receipt telemetry reference가 ledger와 다릅니다")
    expected_smoke_target = build_a100_pretrain_smoke_target(
        cfg, repo_root=root
    )["sha256"]
    expected_receipt_path = (root / SMOKE_ROOT / expected_smoke_target / "receipt.json").absolute()
    if receipt_snapshot.path != expected_receipt_path:
        raise ValueError("A100 smoke receipt는 target prerequisite root의 receipt.json이어야 합니다")
    validate_a100_pretrain_smoke_receipt(
        receipt,
        repo_root=root,
        expected_smoke_target_sha256=expected_smoke_target,
    )
    return ledger


def validate_canonical_pretrain_prerequisites(
    cfg: dict,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """외부 SHA ledger와 모든 referenced evidence를 same-FD로 재검증한다."""

    if str(cfg.get("experiment_role", "")) != "canonical_pretrain":
        return {}
    root = Path(os.path.abspath(Path(repo_root)))
    if cfg.get("campaign_prerequisite") != CANONICAL_PATH:
        raise ValueError(
            f"canonical pretrain campaign_prerequisite는 {CANONICAL_PATH!r}여야 합니다"
        )
    expected_sha = _sha(
        cfg.get("campaign_prerequisite_sha256"),
        label="campaign_prerequisite_sha256",
    )
    ledger_snapshot = snapshot_regular_file(root / CANONICAL_PATH)
    if ledger_snapshot.sha256 != expected_sha:
        raise ValueError("campaign prerequisite ledger가 외부 SHA trust anchor와 다릅니다")
    try:
        payload = json.loads(ledger_snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign prerequisite ledger JSON이 손상됐습니다") from exc
    return validate_canonical_pretrain_ledger_payload(
        cfg,
        payload,
        repo_root=root,
    )


__all__ = [
    "CANONICAL_PATH",
    "SCHEMA_VERSION",
    "validate_canonical_pretrain_ledger_payload",
    "validate_canonical_pretrain_prerequisites",
]
