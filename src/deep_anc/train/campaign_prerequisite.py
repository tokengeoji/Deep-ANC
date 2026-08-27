"""canonical pretrain을 열기 전 campaign 증거 ledger 검증."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from .evaluation_contract import FileSnapshot, snapshot_regular_file
from .experiment_contract import validate_embedded_experiment_contract


SCHEMA_VERSION = 2
CANONICAL_PATH = "results/training_prerequisites/canonical_pretrain.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BASE_PILOTS = {(0.7, 0.5), (0.7, 0.2), (1.0, 0.5), (1.0, 0.2)}
_ALPHA_085 = {(0.85, 0.5), (0.85, 0.2)}


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
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    return target


def _evidence_snapshot(root: Path, item: object, *, label: str) -> FileSnapshot:
    entry = _exact_keys(item, {"path", "sha256"}, label=label)
    snapshot = snapshot_regular_file(_path_inside(root, entry["path"], label=label))
    if snapshot.sha256 != _sha(entry["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 ledger와 다릅니다")
    return snapshot


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
    ledger = _exact_keys(
        payload,
        {
            "schema_version",
            "source",
            "g0",
            "gradient_budget",
            "loss_pilot_selection",
            "measured_probe",
            "a100_smoke_resume",
        },
        label="campaign prerequisite",
    )
    if ledger["schema_version"] != SCHEMA_VERSION:
        raise ValueError("campaign prerequisite schema_version이 다릅니다")

    contract = validate_embedded_experiment_contract(cfg)
    source = _exact_keys(
        ledger["source"],
        {
            "git_commit",
            "source_tree_sha256",
            "bootstrap_receipt_sha256",
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
        "transfer_manifest_sha256": (cfg.get("data") or {}).get(
            "transfer_manifest_sha256"
        ),
        "recorded_transfer_aggregate_sha256": (cfg.get("data") or {}).get(
            "recorded_transfer_aggregate_sha256"
        ),
    }
    if input_generation != expected_generation or not all(expected_generation.values()):
        raise ValueError("campaign prerequisite contract input generation이 불완전합니다")
    expected_source = {
        "git_commit": contract_source.get("git_commit"),
        "source_tree_sha256": contract_source.get("source_tree_sha256"),
        "bootstrap_receipt_sha256": (cfg.get("data") or {}).get(
            "bootstrap_receipt_sha256"
        ),
        "primary_path_sha256": (artifacts.get("primary_path") or {}).get("sha256"),
        "secondary_path_sha256": (artifacts.get("secondary_path") or {}).get("sha256"),
    }
    if source != expected_source or not all(expected_source.values()):
        raise ValueError("campaign prerequisite strict P/S/source/bootstrap identity가 다릅니다")

    g0 = _exact_keys(
        ledger["g0"],
        {"evidence", "all_finite", "nmse_trusted_db", "threshold_exclusive_db"},
        label="campaign G0",
    )
    _evidence_snapshot(root, g0["evidence"], label="campaign G0 evidence")
    nmse = float(g0["nmse_trusted_db"])
    threshold = float(g0["threshold_exclusive_db"])
    if g0["all_finite"] is not True or threshold != -6.0 or not math.isfinite(nmse) or nmse >= threshold:
        raise ValueError("campaign G0는 all-finite이며 trusted NMSE < -6 dB여야 합니다")

    gradient = _exact_keys(
        ledger["gradient_budget"],
        {
            "evidence",
            "strict_ps",
            "lambda_dnh",
            "gradient_share",
            "loss_start_sample",
        },
        label="gradient budget",
    )
    _evidence_snapshot(root, gradient["evidence"], label="gradient budget evidence")
    share = float(gradient["gradient_share"])
    if (
        gradient["strict_ps"] is not True
        or float(gradient["lambda_dnh"]) != float((cfg.get("loss") or {}).get("lambda_dnh"))
        or int(gradient["loss_start_sample"]) != int(cfg.get("loss_start_sample", -1))
        or not math.isfinite(share)
        or not 0.2 <= share <= 0.4
    ):
        raise ValueError(
            "strict-S lambda_dnh gradient share(0.2–0.4) 또는 loss_start_sample이 승인 계약과 다릅니다"
        )

    pilot = _exact_keys(
        ledger["loss_pilot_selection"],
        {"selection_rule", "conditional_alpha_085_triggered", "candidates", "winner"},
        label="loss pilot selection",
    )
    if pilot["selection_rule"] != "minimum_recorded_val_score_db":
        raise ValueError("loss pilot selection rule이 승인 규칙과 다릅니다")
    candidates = pilot["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("loss pilot candidates가 list가 아닙니다")
    rows: list[tuple[tuple[float, float], float]] = []
    for index, raw in enumerate(candidates):
        row = _exact_keys(
            raw,
            {"alpha", "lambda_frame", "run_until_step", "experiment_role", "init_eligible", "score_db", "evidence"},
            label=f"loss pilot candidate[{index}]",
        )
        _evidence_snapshot(root, row["evidence"], label=f"loss pilot evidence[{index}]")
        pair = (float(row["alpha"]), float(row["lambda_frame"]))
        score = float(row["score_db"])
        if (
            int(row["run_until_step"]) != 20_000
            or row["experiment_role"] != "loss_pilot"
            or row["init_eligible"] is not False
            or not math.isfinite(score)
        ):
            raise ValueError("loss pilot candidate 실행 계약이 잘못됐습니다")
        rows.append((pair, score))
    pairs = {pair for pair, _ in rows}
    conditional = pilot["conditional_alpha_085_triggered"] is True
    expected_pairs = _BASE_PILOTS | (_ALPHA_085 if conditional else set())
    if pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("4개 승인 pilot/조건부 alpha=0.85 후보 집합이 다릅니다")
    winner = _exact_keys(
        pilot["winner"], {"alpha", "lambda_frame"}, label="loss pilot winner"
    )
    winner_pair = (float(winner["alpha"]), float(winner["lambda_frame"]))
    expected_winner = min(rows, key=lambda row: (row[1], row[0]))[0]
    cfg_pair = (
        float((cfg.get("loss") or {}).get("nmse_cvar_alpha")),
        float((cfg.get("loss") or {}).get("lambda_frame")),
    )
    if winner_pair != expected_winner or cfg_pair != winner_pair:
        raise ValueError("pilot margin-min winner와 canonical pretrain loss가 다릅니다")

    probe = _exact_keys(
        ledger["measured_probe"],
        {"evidence", "experiment_role", "init_eligible", "run_until_step", "strict_ps", "passed", "alpha", "lambda_frame"},
        label="measured probe",
    )
    _evidence_snapshot(root, probe["evidence"], label="measured probe evidence")
    if (
        probe["experiment_role"] != "measured_probe"
        or probe["init_eligible"] is not False
        or int(probe["run_until_step"]) != 5_000
        or probe["strict_ps"] is not True
        or probe["passed"] is not True
        or (float(probe["alpha"]), float(probe["lambda_frame"])) != winner_pair
    ):
        raise ValueError("winner의 strict-P/S measured 5k probe 증거가 잘못됐습니다")

    smoke = _exact_keys(
        ledger["a100_smoke_resume"],
        {"evidence", "environment_receipt", "device", "cuda", "amp", "stop_step", "resumed_step", "model_equal", "optimizer_equal", "scheduler_equal", "batch_sequence_equal"},
        label="A100 smoke resume",
    )
    _evidence_snapshot(root, smoke["evidence"], label="A100 smoke evidence")
    environment_snapshot = _evidence_snapshot(
        root, smoke["environment_receipt"], label="A100 environment receipt"
    )
    try:
        environment = json.loads(environment_snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("A100 environment receipt JSON이 손상됐습니다") from exc
    devices = environment.get("devices") if isinstance(environment, dict) else None
    if (
        not isinstance(environment, dict)
        or environment.get("cuda_available") is not True
        or environment.get("device_count") != 1
        or not isinstance(devices, list)
        or len(devices) != 1
        or "A100" not in str(devices[0].get("name", ""))
        or environment.get("deterministic_algorithms") is not True
        or environment.get("cudnn_benchmark") is not False
        or environment.get("cudnn_deterministic") is not True
        or environment.get("cublas_workspace_config") not in {":4096:8", ":16:8"}
        or not str(environment.get("torch") or "")
        or not str(environment.get("torch_cuda") or "")
    ):
        raise ValueError(
            "A100 environment receipt가 world1/CUDA/결정론 backend 계약을 증명하지 못합니다"
        )
    stop_step = int(smoke["stop_step"])
    if (
        "A100" not in str(smoke["device"])
        or smoke["cuda"] is not True
        or smoke["amp"] != "bf16"
        or not 200 <= stop_step <= 500
        or int(smoke["resumed_step"]) <= stop_step
        or any(
            smoke[key] is not True
            for key in ("model_equal", "optimizer_equal", "scheduler_equal", "batch_sequence_equal")
        )
    ):
        raise ValueError("A100 bf16 중단→resume 수치등가 증거가 불완전합니다")
    return ledger


__all__ = [
    "CANONICAL_PATH",
    "SCHEMA_VERSION",
    "validate_canonical_pretrain_prerequisites",
]
