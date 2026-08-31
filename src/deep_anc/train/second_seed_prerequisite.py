"""공식 두 번째 seed를 여는 no-replace prerequisite 계약.

두 번째 seed는 첫 seed의 advisory pipeline status로 열지 않는다. 첫 seed의 immutable
recorded-val selection을 다시 분류하고, 그 selection이 가리키는 canonical fine-tune
checkpoint와 canonical pretrain init 계보를 따라가 v7 campaign ledger를 raw evidence부터
재검증한다. test ledger가 한 번이라도 열렸다면 같은 campaign에서 두 번째 seed를 시작할
수 없다.

두 번째 seed의 A100 exact-resume smoke는 seed를 포함한 semantic target이므로 첫 seed의
smoke를 재사용하지 않는다. seed 20260903 target의 receipt/environment/telemetry를 별도로
결속한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any

import torch

from ..config import validate_canonical_training_policy
from .a100_pretrain_smoke import (
    SMOKE_ROOT,
    build_a100_pretrain_smoke_target,
    validate_a100_pretrain_smoke_receipt,
)
from .campaign_prerequisite import (
    CANONICAL_PATH as CANONICAL_CAMPAIGN_PREREQUISITE,
    validate_canonical_pretrain_prerequisites,
)
from .completion_receipt import validate_completion_receipt
from .evaluation_contract import (
    canonical_test_ledger_event_paths_from_payload,
    seed_neutral_campaign_sha256,
    snapshot_regular_file,
    validate_recorded_val_selection,
    validate_test_open_selection,
)
from .experiment_contract import validate_embedded_experiment_contract


SCHEMA_VERSION = 1
KIND = "canonical_second_seed_prerequisite"
PRIMARY_SEED = 20260803
SECONDARY_SEED = 20260903
SECOND_SEED_ROOT = Path("results/training_prerequisites/second_seed")
CROSS_SEED_ROOT = Path("results/finetune_cross_seed")
CONFIG_PATH_KEY = "second_seed_prerequisite"
CONFIG_SHA256_KEY = "second_seed_prerequisite_sha256"
_HEX = frozenset("0123456789abcdef")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 64자리 lowercase SHA-256이 아닙니다")
    return text


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _root(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value)))


def _repo_path(root: Path, value: object, *, label: str) -> Path:
    """lexical/resolved containment와 모든 parent의 non-symlink를 함께 요구한다."""

    lexical_root = _root(root)
    if lexical_root.is_symlink():
        raise ValueError("저장소 root는 symlink일 수 없습니다")
    raw = Path(str(value)).expanduser()
    target = _root(raw if raw.is_absolute() else lexical_root / raw)
    try:
        relative = target.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    cursor = lexical_root
    for component in relative.parts:
        cursor /= component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            # leaf 부재는 뒤의 same-FD snapshot이 명확히 거부한다. 이미 존재하는
            # ancestor까지만 symlink 여부를 검사한다.
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    try:
        target.resolve(strict=False).relative_to(lexical_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}의 resolved path가 저장소 밖입니다: {target}") from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    target = _repo_path(root, path, label=label)
    return target.relative_to(_root(root)).as_posix()


def _snapshot_path(root: Path, value: object, *, label: str):
    return snapshot_regular_file(_repo_path(root, value, label=label))


def _reference(root: Path, value: str | Path, *, label: str) -> dict[str, str]:
    snapshot = _snapshot_path(root, value, label=label)
    return {
        "path": _relative(root, snapshot.path, label=label),
        "sha256": snapshot.sha256,
    }


def _snapshot_ref(root: Path, value: object, *, label: str):
    reference = _exact_keys(value, {"path", "sha256"}, label=label)
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise ValueError(f"{label}.path는 비어 있지 않은 저장소 상대경로여야 합니다")
    snapshot = _snapshot_path(root, raw_path, label=label)
    expected = _sha(reference["sha256"], label=f"{label}.sha256")
    if snapshot.sha256 != expected:
        raise ValueError(f"{label} bytes SHA가 prerequisite와 다릅니다")
    return snapshot


def _json_snapshot(snapshot, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON이 손상됐습니다: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위가 mapping이 아닙니다")
    return payload


def _checkpoint_cfg(snapshot, *, label: str) -> dict[str, Any]:
    try:
        state = torch.load(
            io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} checkpoint를 읽을 수 없습니다") from exc
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError(f"{label} checkpoint에 resolved cfg가 없습니다")
    return state["cfg"]


def validate_second_seed_test_ledger_state(
    selection: dict[str, Any],
    *,
    primary_selection_path: str | Path,
    primary_selection_sha256: str,
    repo_root: str | Path,
) -> str:
    """secondary admission 전 미개봉 또는 legitimate cross 완료만 허용한다."""

    root = _root(repo_root)
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=root
    )
    touched = {
        name: path
        for name, path in event_paths.items()
        if path.exists() or path.is_symlink()
    }
    if not touched:
        return "unopened"
    if set(touched) != {"issued", "running", "completed"}:
        raise ValueError(
            "primary borderline campaign의 test ledger가 이미 열렸습니다. 상태가 "
            "미개봉도 completed cross-seed도 "
            f"아닙니다: {sorted(touched)}"
        )
    campaign = _sha(
        selection.get("seed_neutral_campaign_sha256"),
        label="primary seed-neutral campaign SHA",
    )
    final_path = root / CROSS_SEED_ROOT / campaign / "recorded_val_selection.json"
    final_snapshot = snapshot_regular_file(final_path)
    final_payload = _json_snapshot(final_snapshot, label="cross-seed final selection")
    validate_test_open_selection(final_payload, repo_root=root)
    rows = final_payload.get("seed_selections")
    if not isinstance(rows, list):
        raise ValueError("cross-seed final selection에 seed selection refs가 없습니다")
    expected_path = _repo_path(
        root, primary_selection_path, label="primary selection path"
    ).absolute()
    expected_sha = _sha(
        primary_selection_sha256, label="primary selection SHA"
    )
    if not any(
        isinstance(row, dict)
        and row.get("seed") == PRIMARY_SEED
        and Path(str(row.get("path", ""))).absolute() == expected_path
        and row.get("sha256") == expected_sha
        for row in rows
    ):
        raise ValueError("cross-seed final이 현재 sealed primary selection을 참조하지 않습니다")
    return "cross_seed_completed"


def second_seed_prerequisite_path(
    seed_neutral_sha256: object, *, repo_root: str | Path
) -> Path:
    """한 seed-neutral campaign에 유일한 secondary prerequisite pathname."""

    root = _root(repo_root)
    campaign = _sha(seed_neutral_sha256, label="primary seed-neutral campaign SHA")
    return _repo_path(
        root,
        root / SECOND_SEED_ROOT / campaign / f"seed_{SECONDARY_SEED}.json",
        label="second-seed prerequisite fixed path",
    )


def _validate_primary_chain(
    secondary_cfg: dict[str, Any],
    selection: dict[str, Any],
    *,
    root: Path,
    selection_path: Path,
    selection_sha256: str,
) -> dict[str, Any]:
    """primary selection에서 실제 50k→100k cfg를 역추적해 raw authority를 닫는다."""

    decision = validate_recorded_val_selection(selection, repo_root=root)
    if selection.get("seed") != PRIMARY_SEED or decision.get("status") != "borderline":
        raise ValueError(
            "second seed는 seed 20260803의 numeric/CI borderline selection만 엽니다"
        )
    if selection.get("decision") != decision:
        raise ValueError("primary selection top-level decision이 raw 재분류와 다릅니다")

    test_ledger_state = validate_second_seed_test_ledger_state(
        selection,
        primary_selection_path=selection_path,
        primary_selection_sha256=selection_sha256,
        repo_root=root,
    )

    selected = selection.get("selected")
    if not isinstance(selected, dict):  # canonical selection validator의 방어 반복
        raise ValueError("primary selection.selected가 mapping이 아닙니다")
    fine_snapshot = _snapshot_path(
        root, selected.get("checkpoint"), label="primary fine-tune checkpoint"
    )
    if fine_snapshot.sha256 != selected.get("checkpoint_sha256"):
        raise ValueError("primary fine-tune checkpoint SHA가 selection과 다릅니다")
    fine_cfg = _checkpoint_cfg(fine_snapshot, label="primary fine-tune")
    fine_contract = validate_embedded_experiment_contract(fine_cfg)
    validate_canonical_training_policy(fine_cfg)
    if (
        fine_cfg.get("experiment_role") != "canonical_finetune"
        or fine_cfg.get("seed") != PRIMARY_SEED
        or fine_cfg.get("init_eligible") is not False
    ):
        raise ValueError("primary selection checkpoint가 공식 seed 20260803 fine-tune이 아닙니다")
    fine_completion = validate_completion_receipt(
        fine_snapshot.path.parent,
        expected_role="canonical_finetune",
        expected_init_eligible=False,
        repo_root=root,
    )
    if fine_completion.get("experiment_contract_sha256") != fine_contract.get("sha256"):
        raise ValueError("primary fine-tune completion contract가 checkpoint와 다릅니다")

    init_value = fine_cfg.get("init_ckpt")
    if not isinstance(init_value, str) or not init_value.strip():
        raise ValueError("primary fine-tune checkpoint에 canonical init_ckpt가 없습니다")
    pretrain_snapshot = _snapshot_path(
        root, init_value, label="primary canonical pretrain init checkpoint"
    )
    if pretrain_snapshot.path.name != "best.pt":
        raise ValueError("primary fine-tune init은 canonical pretrain best.pt여야 합니다")
    pretrain_cfg = _checkpoint_cfg(pretrain_snapshot, label="primary canonical pretrain")
    pretrain_contract = validate_embedded_experiment_contract(pretrain_cfg)
    validate_canonical_training_policy(pretrain_cfg)
    if (
        pretrain_cfg.get("experiment_role") != "canonical_pretrain"
        or pretrain_cfg.get("seed") != PRIMARY_SEED
        or pretrain_cfg.get("init_eligible") is not True
    ):
        raise ValueError("primary init checkpoint가 공식 seed 20260803 pretrain이 아닙니다")
    pretrain_completion = validate_completion_receipt(
        pretrain_snapshot.path.parent,
        expected_role="canonical_pretrain",
        expected_init_eligible=True,
        repo_root=root,
    )
    if pretrain_completion.get("experiment_contract_sha256") != pretrain_contract.get(
        "sha256"
    ):
        raise ValueError("primary pretrain completion contract가 checkpoint와 다릅니다")

    # 바로 이 저장 cfg가 seed=20260803 canonical view다. 수기로 seed만 바꾼 cfg를
    # 만들지 않고 기존 v7 validator가 raw G0/pilot/probe/gradient/smoke를 전부 다시 연다.
    validate_canonical_pretrain_prerequisites(pretrain_cfg, repo_root=root)

    primary_neutral = seed_neutral_campaign_sha256(pretrain_cfg)
    secondary_neutral = seed_neutral_campaign_sha256(secondary_cfg)
    if primary_neutral != secondary_neutral:
        raise ValueError(
            "primary/secondary canonical pretrain seed-neutral digest가 다릅니다: "
            f"primary={primary_neutral}, secondary={secondary_neutral}"
        )
    return {
        "decision": decision,
        "fine_cfg": fine_cfg,
        "pretrain_cfg": pretrain_cfg,
        "pretrain_neutral_sha256": primary_neutral,
        "test_ledger_state": test_ledger_state,
    }


def _validate_secondary_smoke(
    secondary_cfg: dict[str, Any],
    smoke: dict[str, Any],
    *,
    root: Path,
) -> str:
    expected_target = _sha(
        build_a100_pretrain_smoke_target(secondary_cfg, repo_root=root).get("sha256"),
        label="secondary A100 smoke target",
    )
    declared_target = _sha(smoke["target_sha256"], label="declared secondary smoke target")
    if declared_target != expected_target:
        raise ValueError("secondary A100 smoke target이 seed 20260903 config와 다릅니다")

    receipt_snapshot = _snapshot_ref(root, smoke["evidence"], label="secondary smoke receipt")
    environment_snapshot = _snapshot_ref(
        root, smoke["environment_receipt"], label="secondary smoke environment receipt"
    )
    telemetry_snapshot = _snapshot_ref(
        root, smoke["telemetry"], label="secondary smoke telemetry"
    )
    target_root = root / SMOKE_ROOT / expected_target
    expected_paths = {
        receipt_snapshot.path: target_root / "receipt.json",
        environment_snapshot.path: target_root / "environment_receipt.json",
        telemetry_snapshot.path: target_root / "telemetry.json",
    }
    for actual, expected in expected_paths.items():
        if actual != expected.absolute():
            raise ValueError(
                "secondary smoke artifact가 seed-specific target fixed path가 아닙니다: "
                f"actual={actual}, expected={expected}"
            )
    receipt = _json_snapshot(receipt_snapshot, label="secondary smoke receipt")
    if receipt.get("environment_receipt") != smoke["environment_receipt"]:
        raise ValueError("secondary smoke environment reference가 receipt와 다릅니다")
    if receipt.get("telemetry") != smoke["telemetry"]:
        raise ValueError("secondary smoke telemetry reference가 receipt와 다릅니다")
    validate_a100_pretrain_smoke_receipt(
        receipt,
        repo_root=root,
        expected_smoke_target_sha256=expected_target,
    )
    return expected_target


def validate_second_seed_prerequisite_payload(
    secondary_cfg: dict[str, Any],
    payload: object,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """prospective/published payload를 같은 raw validator로 검증한다."""

    root = _root(repo_root)
    prerequisite = _exact_keys(
        payload,
        {"schema_version", "kind", "shared", "primary", "secondary"},
        label="second-seed prerequisite",
    )
    if prerequisite["schema_version"] != SCHEMA_VERSION or prerequisite["kind"] != KIND:
        raise ValueError("second-seed prerequisite schema/kind가 다릅니다")
    shared = _exact_keys(
        prerequisite["shared"],
        {
            "campaign_prerequisite",
            "bootstrap_receipt_sha256",
            "loss_selection_sha256",
        },
        label="second-seed shared identity",
    )
    primary = _exact_keys(
        prerequisite["primary"],
        {"seed", "recorded_val_selection", "seed_neutral_campaign_sha256"},
        label="second-seed primary authority",
    )
    secondary = _exact_keys(
        prerequisite["secondary"],
        {"seed", "a100_smoke_resume"},
        label="second-seed secondary authority",
    )
    smoke = _exact_keys(
        secondary["a100_smoke_resume"],
        {
            "target_sha256",
            "evidence",
            "environment_receipt",
            "telemetry",
        },
        label="second-seed A100 smoke",
    )
    if primary["seed"] != PRIMARY_SEED or secondary["seed"] != SECONDARY_SEED:
        raise ValueError("second-seed prerequisite 공식 seed pair가 다릅니다")
    if (
        secondary_cfg.get("experiment_role") != "canonical_pretrain"
        or secondary_cfg.get("seed") != SECONDARY_SEED
        or secondary_cfg.get("init_eligible") is not True
    ):
        raise ValueError("second-seed target config가 공식 seed 20260903 pretrain이 아닙니다")
    validate_embedded_experiment_contract(secondary_cfg)
    validate_canonical_training_policy(secondary_cfg)

    ledger_snapshot = _snapshot_ref(
        root, shared["campaign_prerequisite"], label="primary v7 campaign prerequisite"
    )
    if ledger_snapshot.path != (root / CANONICAL_CAMPAIGN_PREREQUISITE).absolute():
        raise ValueError("second seed는 canonical_pretrain.json v7 ledger만 재사용합니다")
    shared_bootstrap = _sha(
        shared["bootstrap_receipt_sha256"], label="shared bootstrap receipt SHA"
    )
    shared_loss = _sha(shared["loss_selection_sha256"], label="shared loss selection SHA")

    selection_snapshot = _snapshot_ref(
        root, primary["recorded_val_selection"], label="primary borderline selection"
    )
    selection = _json_snapshot(selection_snapshot, label="primary borderline selection")
    declared_campaign = _sha(
        primary["seed_neutral_campaign_sha256"],
        label="primary fine-tune seed-neutral campaign SHA",
    )
    if selection.get("seed_neutral_campaign_sha256") != declared_campaign:
        raise ValueError("primary selection seed-neutral digest가 prerequisite와 다릅니다")

    chain = _validate_primary_chain(
        secondary_cfg,
        selection,
        root=root,
        selection_path=selection_snapshot.path,
        selection_sha256=selection_snapshot.sha256,
    )
    primary_pretrain = chain["pretrain_cfg"]
    primary_finetune = chain["fine_cfg"]
    for label, cfg in (
        ("primary pretrain", primary_pretrain),
        ("secondary pretrain", secondary_cfg),
    ):
        if cfg.get("campaign_prerequisite") != CANONICAL_CAMPAIGN_PREREQUISITE:
            raise ValueError(f"{label} campaign prerequisite path가 canonical이 아닙니다")
        if cfg.get("campaign_prerequisite_sha256") != ledger_snapshot.sha256:
            raise ValueError(f"{label} campaign prerequisite SHA가 shared ledger와 다릅니다")
        data = cfg.get("data")
        if not isinstance(data, dict) or data.get("bootstrap_receipt_sha256") != shared_bootstrap:
            raise ValueError(f"{label} bootstrap receipt SHA가 shared identity와 다릅니다")
        if cfg.get("loss_selection_sha256") != shared_loss:
            raise ValueError(f"{label} loss selection SHA가 shared identity와 다릅니다")
    if primary_finetune.get("loss_selection_sha256") != shared_loss:
        raise ValueError("primary fine-tune loss selection이 pretrain winner와 다릅니다")

    _validate_secondary_smoke(secondary_cfg, smoke, root=root)
    return prerequisite


def build_second_seed_prerequisite_payload(
    secondary_cfg: dict[str, Any],
    *,
    primary_selection: str | Path,
    smoke_receipt: str | Path,
    smoke_environment_receipt: str | Path,
    smoke_telemetry: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """raw artifact reference만 받아 prospective payload를 만들고 즉시 재검증한다."""

    root = _root(repo_root)
    selection_ref = _reference(
        root, primary_selection, label="primary borderline selection"
    )
    selection = _json_snapshot(
        _snapshot_ref(root, selection_ref, label="primary borderline selection"),
        label="primary borderline selection",
    )
    campaign = _sha(
        selection.get("seed_neutral_campaign_sha256"),
        label="primary selection seed-neutral campaign SHA",
    )
    target = _sha(
        build_a100_pretrain_smoke_target(secondary_cfg, repo_root=root).get("sha256"),
        label="secondary A100 smoke target",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "shared": {
            "campaign_prerequisite": _reference(
                root,
                root / CANONICAL_CAMPAIGN_PREREQUISITE,
                label="primary v7 campaign prerequisite",
            ),
            "bootstrap_receipt_sha256": str(
                (secondary_cfg.get("data") or {}).get("bootstrap_receipt_sha256", "")
            ),
            "loss_selection_sha256": str(
                secondary_cfg.get("loss_selection_sha256", "")
            ),
        },
        "primary": {
            "seed": PRIMARY_SEED,
            "recorded_val_selection": selection_ref,
            "seed_neutral_campaign_sha256": campaign,
        },
        "secondary": {
            "seed": SECONDARY_SEED,
            "a100_smoke_resume": {
                "target_sha256": target,
                "evidence": _reference(
                    root, smoke_receipt, label="secondary smoke receipt"
                ),
                "environment_receipt": _reference(
                    root,
                    smoke_environment_receipt,
                    label="secondary smoke environment receipt",
                ),
                "telemetry": _reference(
                    root, smoke_telemetry, label="secondary smoke telemetry"
                ),
            },
        },
    }
    validate_second_seed_prerequisite_payload(
        secondary_cfg, payload, repo_root=root
    )
    return payload


def validate_second_seed_prerequisites(
    cfg: dict[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """resolved secondary cfg의 외부 SHA와 fixed path에서 published payload를 검증한다."""

    if cfg.get("experiment_role") != "canonical_pretrain" or cfg.get("seed") != SECONDARY_SEED:
        return {}
    root = _root(repo_root)
    configured = cfg.get(CONFIG_PATH_KEY)
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError(f"seed {SECONDARY_SEED}에는 {CONFIG_PATH_KEY}가 필요합니다")
    snapshot = _snapshot_path(root, configured, label="second-seed prerequisite")
    expected_sha = _sha(cfg.get(CONFIG_SHA256_KEY), label=CONFIG_SHA256_KEY)
    if snapshot.sha256 != expected_sha:
        raise ValueError("second-seed prerequisite bytes가 외부 SHA trust anchor와 다릅니다")
    payload = _json_snapshot(snapshot, label="second-seed prerequisite")
    primary = payload.get("primary") if isinstance(payload, dict) else None
    campaign = primary.get("seed_neutral_campaign_sha256") if isinstance(primary, dict) else None
    expected_path = second_seed_prerequisite_path(campaign, repo_root=root)
    if snapshot.path != expected_path.absolute():
        raise ValueError(
            "second-seed prerequisite가 seed-neutral campaign fixed path가 아닙니다: "
            f"actual={snapshot.path}, expected={expected_path}"
        )
    return validate_second_seed_prerequisite_payload(cfg, payload, repo_root=root)


def prerequisite_json_bytes(payload: dict[str, Any]) -> bytes:
    """exclusive JSON writer와 byte-for-byte 같은 prospective SHA 입력."""

    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def prerequisite_sha256(payload: dict[str, Any]) -> str:
    return _sha256_bytes(prerequisite_json_bytes(payload))


__all__ = [
    "CONFIG_PATH_KEY",
    "CONFIG_SHA256_KEY",
    "KIND",
    "PRIMARY_SEED",
    "SCHEMA_VERSION",
    "SECONDARY_SEED",
    "SECOND_SEED_ROOT",
    "build_second_seed_prerequisite_payload",
    "prerequisite_json_bytes",
    "prerequisite_sha256",
    "second_seed_prerequisite_path",
    "validate_second_seed_prerequisite_payload",
    "validate_second_seed_prerequisites",
    "validate_second_seed_test_ledger_state",
]
