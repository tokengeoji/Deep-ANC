#!/usr/bin/env python3
"""준비 게이트→fine-tune→독립 val/test→완료 게이트를 순차 실행한다.

오디오 장치는 열지 않는다. P/S·recorded·완료된 pretrain 이 준비되지 않았으면 첫
게이트에서 종료하며 **``runs/`` 아래에 아무것도 만들지 않는다** — 학습 디렉터리의
존재가 "학습이 실제로 시작됐다"는 의미를 유지해야 하기 때문이다. 감사·상태 산출물은
전부 ``results/finetune_autostart/<run-key>/`` 로 간다.

중단 뒤 ``last.pt`` 가 있더라도 자동 재개하지 않는다. 동일 experiment contract가
checkpoint 안에 증명된 경우에만 ``--resume``을 명시해 같은 run을 재개할 수 있고,
``best.pt``만 남은 모호한 상태도 덮어쓰지 않는다. 재개 판단은 status.json이 아니라
checkpoint의 계약과 완전한 optimizer/scheduler/RNG/data-stream 상태를 사용한다.

  .venv/bin/python scripts/train/run_finetune_pipeline.py \
    --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured

  --check-only   준비 리포트만 생성 (학습 미시작)
  --status       lock 없이 현재 상태만 출력

종료 코드: 0 OK / 1 NOT READY / 2 config·모호한 재개 / 3 다른 프로세스가 같은 run 을
학습 중(train.lock) / 4 pipeline 중복 실행(pipeline.lock) / 5 단계 실패
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_train_config  # noqa: E402
from deep_anc.train.finetune_readiness import (  # noqa: E402
    audit_finetune_readiness,
    render_audit_markdown,
)
from deep_anc.train.evaluation_contract import (  # noqa: E402
    CAPABILITY_ENV,
    canonical_test_ledger_paths,
    issue_test_capability,
    publish_directory_noreplace,
    read_json_snapshot,
    snapshot_regular_file,
    write_json_exclusive,
)
from deep_anc.train.completion_receipt import validate_completion_receipt  # noqa: E402
from deep_anc.train.experiment_contract import (  # noqa: E402
    validate_embedded_experiment_contract,
)
from deep_anc.train.pipeline_status import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_PIPELINE_LOCKED,
    EXIT_STAGE_FAILED,
    EXIT_TRAIN_LOCKED,
    PipelineStatus,
    atomic_write_text,
    config_fingerprint,
    read_status,
    sha256_text,
)
from deep_anc.train.process_lock import (  # noqa: E402
    LockHeldError,
    ProcessLock,
    autostart_state_dir,
    finetune_run_key,
    resolve_run_dir,
)


class StepFailed(RuntimeError):
    """자식 단계 실패. 어느 단계에서 어떤 코드로 죽었는지를 함께 나른다.

    returncode 만으로는 "train.py 의 3(중복 학습)"과 "다른 자식의 우연한 3"을 구분할 수
    없어서 단계 이름을 반드시 같이 싣는다.
    """

    def __init__(self, step: str, returncode: int) -> None:
        super().__init__(f"{step} 단계가 exit {returncode}로 실패했습니다")
        self.step = step
        self.returncode = returncode


def _repo_state_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"state-dir은 Deep_ANC 내부여야 합니다: {path}") from exc
    return path


def _resolved_input_path(value: str | Path) -> Path:
    """부모가 감사한 바로 그 파일의 절대경로.

    상대 경로를 그대로 자식에게 넘기면, 저장소 밖 CWD 에서 실행했을 때 부모는
    ``$CWD/configs/...`` 를 감사하고 자식 train.py 는 ``$REPO/configs/...`` 를 학습한다.
    감사한 설정과 학습한 설정이 달라지는 fail-open 이라 반드시 해석해서 넘긴다.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = Path.cwd() / path
    return (cwd_candidate if cwd_candidate.exists() else REPO_ROOT / path).resolve()


def _run(
    command: list[str],
    *,
    step: str,
    status: PipelineStatus,
    environment: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(command), flush=True)
    started = time.monotonic()
    child_environment = os.environ.copy()
    if environment:
        child_environment.update(environment)
    completed = subprocess.run(  # noqa: S603
        command, cwd=REPO_ROOT, check=False, env=child_environment
    )
    status.record_step(step, command, completed.returncode, time.monotonic() - started)
    if completed.returncode != 0:
        raise StepFailed(step, completed.returncode)


def _train_lock_busy(train_lock: Path) -> bool:
    """자식이 잡을 train.lock 이 이미 물려 있는지 미리 본다.

    이건 최적화일 뿐 권한이 아니다(TOCTOU 가 있다). 실제 배제는 자식 rank0 의 flock 이
    하고 우리는 그 exit 3 을 그대로 3으로 매핑한다. 여기서 미리 걸러내는 이유는 무거운
    recorded 전수 QA 를 헛돌리지 않기 위해서다.
    """

    if not train_lock.exists():
        return False
    try:
        ProcessLock(train_lock, role="probe").acquire().release()
    except LockHeldError:
        return True
    except OSError:
        return False
    return False


def _warn_legacy_run_audit(run_dir: Path) -> None:
    """옛 코드가 학습 디렉터리에 남긴 감사 산출물을 알린다(삭제하지 않는다)."""

    legacy = run_dir / "audit" / "readiness.json"
    if legacy.exists() and not (run_dir / "ckpt").exists():
        print(
            f"[주의] 학습이 시작된 적 없는데 감사 산출물이 남아 있습니다: {legacy}\n"
            f"       구버전 파이프라인의 잔재입니다. 보존이 필요하면 "
            f"results/finetune_autostart/<run-key>/audit/legacy/ 로 옮기세요.",
            file=sys.stderr,
        )


def _print_status(status_path: Path, pipeline_lock: Path, train_lock: Path) -> None:
    payload = read_status(status_path)
    print(f"status: {status_path}")
    if payload is None:
        print("  (없음 — 아직 실행된 적이 없습니다)")
    else:
        print(f"  phase={payload.get('phase')} exit_code={payload.get('exit_code')}")
        print(f"  mode={payload.get('mode')} run_dir={payload.get('run_dir')}")
        readiness = payload.get("readiness") or {}
        if readiness:
            print(f"  readiness.ok={readiness.get('ok')} 실패={readiness.get('failed_checks')}")
        for step in payload.get("steps") or []:
            print(f"    - {step['name']}: exit {step['returncode']} ({step['duration_seconds']}s)")
    for label, path in (("pipeline", pipeline_lock), ("train", train_lock)):
        held = _train_lock_busy(path)
        print(f"  {label}.lock 보유중={held} ({path})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_finetune.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "명시적으로 재개할 last.pt. 생략했는데 ckpt_dir에 last.pt가 있으면 구형 "
            "artifact 자동 재개를 막기 위해 중단"
        ),
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="key=value override"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="준비 리포트만 생성하고 학습은 시작하지 않음 (같은 pipeline.lock 으로 보호됨)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "감사·상태 산출 경로(기본: results/finetune_autostart/<run-key>). "
            "lock 은 중복 실행 우회를 막기 위해 항상 기본 경로에 둔다. Deep_ANC 내부만 허용"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="lock 없이 현재 status.json 과 lock 보유자만 출력하고 종료",
    )
    parser.add_argument(
        "--cross-seed-selection",
        action="append",
        default=[],
        help="2-seed finalize 전용 val selection.json (정확히 두 번 지정)",
    )
    parser.add_argument(
        "--cross-seed-final-selection",
        default=None,
        help="2-seed winner selection no-replace 출력 경로",
    )
    return parser


def _resolve_explicit_resume(requested: str | None, ckpt_dir: Path) -> Path | None:
    """디스크에 last.pt가 있다는 이유만으로 실험을 자동 재개하지 않는다."""

    existing_last = ckpt_dir / "last.pt"
    if requested is None:
        if existing_last.exists():
            raise ValueError(
                f"기존 last.pt가 있지만 --resume이 명시되지 않았습니다: {existing_last}. "
                "새 실험이면 신규 ckpt_dir을 쓰고, 같은 실험 재개라면 --resume을 "
                "명시하세요 (experiment contract가 추가 검증합니다)"
            )
        return None
    path = Path(requested).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"명시한 resume checkpoint가 없습니다: {path}")
    if path.name != "last.pt":
        raise ValueError(f"resume은 best.pt가 아니라 last.pt여야 합니다: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


VAL_BORDERLINE_MARGIN_DB = 0.3
OFFICIAL_FINETUNE_SEEDS = frozenset({20260803, 20260903})
_CAMPAIGN_OPERATIONAL_KEYS = {
    "seed",
    "resume",
    "run_until_step",
    "ckpt_dir",
    "resolved_contract_run_dir",
    "experiment_contract",
    "experiment_contract_sha256",
}


def seed_neutral_campaign_sha256(
    cfg: dict, *, _visited_init_sha: frozenset[str] = frozenset()
) -> str:
    """seed/run 위치만 제거하고 init 계보까지 의미로 결속한 campaign digest."""

    contract = validate_embedded_experiment_contract(cfg)
    semantic = copy.deepcopy(
        {
            key: value
            for key, value in cfg.items()
            if key not in _CAMPAIGN_OPERATIONAL_KEYS
        }
    )
    init_value = cfg.get("init_ckpt")
    if init_value:
        init_path = Path(str(init_value)).expanduser()
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        init_snapshot = snapshot_regular_file(init_path)
        if init_snapshot.sha256 in _visited_init_sha:
            raise ValueError("campaign init checkpoint 계보에 순환이 있습니다")
        init_state = torch.load(
            io.BytesIO(init_snapshot.content), map_location="cpu", weights_only=False
        )
        if not isinstance(init_state, dict) or not isinstance(
            init_state.get("cfg"), dict
        ):
            raise ValueError("campaign init checkpoint에 resolved cfg가 없습니다")
        init_cfg = init_state["cfg"]
        semantic["init_ckpt"] = {
            "experiment_role": init_cfg.get("experiment_role"),
            "init_eligible": init_cfg.get("init_eligible"),
            "loss_selection_sha256": init_cfg.get("loss_selection_sha256"),
            "seed_neutral_campaign_sha256": seed_neutral_campaign_sha256(
                init_cfg,
                _visited_init_sha=_visited_init_sha | {init_snapshot.sha256},
            ),
        }
    source = contract.get("source") or {}
    artifact_identity = {
        name: {
            key: entry.get(key)
            for key in ("exists", "size_bytes", "sha256")
            if key in entry
        }
        for name, entry in sorted((contract.get("artifacts") or {}).items())
        if name != "init_checkpoint" and isinstance(entry, dict)
    }
    payload = {
        "schema_version": 1,
        "semantic_config": semantic,
        "source": {
            "git_commit": source.get("git_commit"),
            "source_tree_sha256": source.get("source_tree_sha256"),
        },
        "artifacts": artifact_identity,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in data.files:
        raise ValueError(f"recorded-val G4 필드가 없습니다: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"recorded-val G4 {key}는 scalar여야 합니다")
    return value.reshape(-1)[0].item()


def classify_recorded_val_metrics(metrics_bytes: bytes) -> dict:
    """모든 G4/CI 경계의 최소 여유로 clear-pass/fail/borderline을 판정한다."""

    try:
        archive = np.load(io.BytesIO(metrics_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("recorded-val metrics.npz가 손상됐습니다") from exc
    with archive as data:
        verdict = str(_npz_scalar(data, "g4_verdict"))
        threshold = float(_npz_scalar(data, "g4_max_out_of_band_amplification_db"))
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("G4 do-no-harm threshold가 유효하지 않습니다")
        margins = {
            "trusted_mean_db": -float(_npz_scalar(data, "nmse_trusted_mean_db")),
            "fullband_mean_db": -float(_npz_scalar(data, "nmse_fullband_mean_db")),
            "worst_source_mean_db": -float(
                _npz_scalar(data, "g4_worst_source_trusted_mean_db")
            ),
            "worst_source_worst10_db": -float(
                _npz_scalar(data, "g4_worst_source_trusted_worst10_db")
            ),
            "do_no_harm_db": float(
                _npz_scalar(data, "g4_worst_octave_worst10_db")
            )
            + threshold,
        }
        if "source_trusted_ci_hi_db" not in data.files:
            raise ValueError("recorded-val bootstrap CI 상단이 없습니다")
        ci_hi = np.asarray(data["source_trusted_ci_hi_db"], dtype=np.float64)
        if ci_hi.size == 0 or not bool(np.isfinite(ci_hi).all()):
            raise ValueError("recorded-val bootstrap CI 상단이 finite/nonempty가 아닙니다")
        margins["worst_source_ci_hi_db"] = -float(np.max(ci_hi))
        if not all(math.isfinite(value) for value in margins.values()):
            raise ValueError("recorded-val G4 margin에 NaN/Inf가 있습니다")
        flags = {
            key: bool(_npz_scalar(data, key))
            for key in (
                "g4_trusted_pass",
                "g4_fullband_pass",
                "g4_source_pass",
                "g4_do_no_harm_pass",
                "g4_power_pass",
                "g4_ci_pass",
                "g4_pass",
            )
        }
        selection_metric = float(
            _npz_scalar(data, "nmse_trusted_worst10_mean_db")
        )
    if not math.isfinite(selection_metric):
        raise ValueError("recorded-val 선택 지표가 non-finite")
    minimum = min(margins.values())
    near_boundary = any(
        abs(value) <= VAL_BORDERLINE_MARGIN_DB for value in margins.values()
    )
    numeric_pass = all(value >= 0.0 for value in margins.values())
    discrete_pass = all(flags.values()) and verdict == "PASS"
    if verdict == "INCONCLUSIVE" or near_boundary:
        status = "borderline"
    elif numeric_pass and discrete_pass:
        status = "clear_pass"
    else:
        status = "clear_fail"
    return {
        "status": status,
        "boundary_margin_db": VAL_BORDERLINE_MARGIN_DB,
        "minimum_margin_db": minimum,
        "margins_db": margins,
        "g4_verdict": verdict,
        "g4_flags": flags,
        "selection_metric_db": selection_metric,
    }


def finalize_cross_seed_selection(
    *, selection_paths: list[Path], final_selection_path: Path
) -> dict:
    """공식 두 seed의 val-only bundle에서 G4 PASS 최소여유 최대 모델을 고정한다."""

    if len(selection_paths) != 2:
        raise ValueError("cross-seed finalize에는 두 selection bundle이 필요합니다")
    bundles: list[dict] = []
    seeds: set[int] = set()
    campaigns: set[str] = set()
    for path in selection_paths:
        payload, snapshot = read_json_snapshot(path)
        selected = payload.get("selected")
        decision = selected.get("decision") if isinstance(selected, dict) else None
        seed = payload.get("seed")
        if (
            payload.get("selection_split") != "val"
            or not isinstance(selected, dict)
            or not isinstance(decision, dict)
            or not isinstance(seed, int)
        ):
            raise ValueError(f"cross-seed selection bundle 구조가 잘못됐습니다: {path}")
        if seed not in OFFICIAL_FINETUNE_SEEDS or seed in seeds:
            raise ValueError(f"cross-seed 공식 seed가 중복/범위 밖입니다: {seed}")
        checkpoint_snapshot = snapshot_regular_file(selected.get("checkpoint", ""))
        manifest_snapshot = snapshot_regular_file(payload.get("manifest", ""))
        metrics_snapshot = snapshot_regular_file(
            Path(str(selected.get("evaluation_dir", ""))) / "metrics.npz"
        )
        if checkpoint_snapshot.sha256 != selected.get("checkpoint_sha256"):
            raise ValueError("cross-seed checkpoint bytes가 selection 뒤 바뀌었습니다")
        if manifest_snapshot.sha256 != payload.get("manifest_sha256"):
            raise ValueError("cross-seed manifest bytes가 selection 뒤 바뀌었습니다")
        if metrics_snapshot.sha256 != selected.get("metrics_sha256"):
            raise ValueError("cross-seed val metrics bytes가 selection 뒤 바뀌었습니다")
        state = torch.load(
            io.BytesIO(checkpoint_snapshot.content), map_location="cpu", weights_only=False
        )
        if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
            raise ValueError("cross-seed checkpoint에 resolved cfg가 없습니다")
        saved_cfg = state["cfg"]
        embedded = validate_embedded_experiment_contract(saved_cfg)
        if embedded["sha256"] != payload.get("experiment_contract_sha256"):
            raise ValueError("cross-seed embedded contract가 selection과 다릅니다")
        if saved_cfg.get("seed") != seed or selected.get("seed") != seed:
            raise ValueError("cross-seed selection/checkpoint seed가 다릅니다")
        calculated_campaign = seed_neutral_campaign_sha256(saved_cfg)
        declared_campaign = str(payload.get("seed_neutral_campaign_sha256", ""))
        if calculated_campaign != declared_campaign or selected.get(
            "seed_neutral_campaign_sha256"
        ) != declared_campaign:
            raise ValueError("cross-seed seed-neutral campaign digest가 손상됐습니다")
        current_decision = classify_recorded_val_metrics(metrics_snapshot.content)
        if decision != current_decision or payload.get("decision") != current_decision:
            raise ValueError("cross-seed val decision이 metrics bytes와 다릅니다")
        with np.load(io.BytesIO(metrics_snapshot.content), allow_pickle=False) as data:
            evaluated = {
                "split": str(_npz_scalar(data, "split")),
                "checkpoint_sha256": str(_npz_scalar(data, "checkpoint_sha256")),
                "manifest_sha256": str(_npz_scalar(data, "manifest_sha256")),
                "experiment_contract_sha256": str(
                    _npz_scalar(data, "experiment_contract_sha256")
                ),
            }
        if evaluated != {
            "split": "val",
            "checkpoint_sha256": checkpoint_snapshot.sha256,
            "manifest_sha256": manifest_snapshot.sha256,
            "experiment_contract_sha256": embedded["sha256"],
        }:
            raise ValueError("cross-seed val metrics provenance가 selection과 다릅니다")
        seeds.add(seed)
        campaigns.add(declared_campaign)
        bundles.append(
            {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
                "seed": seed,
                "payload": payload,
                "checkpoint_snapshot": checkpoint_snapshot,
                "manifest_snapshot": manifest_snapshot,
                "metrics_snapshot": metrics_snapshot,
                "state": state,
            }
        )
    if seeds != set(OFFICIAL_FINETUNE_SEEDS):
        raise ValueError(
            f"cross-seed selection은 {sorted(OFFICIAL_FINETUNE_SEEDS)}가 모두 필요합니다"
        )
    if len(campaigns) != 1:
        raise ValueError("두 seed selection이 같은 seed-neutral campaign이 아닙니다")
    first_seed = next(item for item in bundles if item["seed"] == 20260803)
    if first_seed["payload"]["selected"]["decision"].get("status") != "borderline":
        raise ValueError(
            "seed 20260803이 borderline/INCONCLUSIVE로 second-seed 조건을 충족하지 않았습니다"
        )
    eligible = [
        item
        for item in bundles
        if item["payload"]["selected"]["decision"].get("g4_verdict") == "PASS"
        and float(
            item["payload"]["selected"]["decision"].get(
                "minimum_margin_db", float("-inf")
            )
        )
        >= 0.0
    ]
    if not eligible:
        raise ValueError("두 seed 중 val G4 PASS인 checkpoint가 없습니다 — test를 열 수 없습니다")
    winner = max(
        eligible,
        key=lambda item: (
            float(item["payload"]["selected"]["decision"]["minimum_margin_db"]),
            -int(item["seed"]),
        ),
    )
    source = winner["payload"]
    selected_source = source["selected"]
    payload = {
        "schema_version": 2,
        "selection_split": "val",
        "test_opened": False,
        "manifest": source["manifest"],
        "manifest_sha256": source["manifest_sha256"],
        "experiment_contract_sha256": source["experiment_contract_sha256"],
        "seed_neutral_campaign_sha256": next(iter(campaigns)),
        "seed": int(winner["seed"]),
        "selected": source["selected"],
        "decision": {
            "status": "cross_seed_final",
            "selection_rule": "val_g4_pass_then_maximum_minimum_margin_db",
            "selected_seed": int(winner["seed"]),
        },
        "seed_selections": [
            {key: item[key] for key in ("path", "sha256", "seed")}
            for item in sorted(bundles, key=lambda item: item["seed"])
        ],
    }
    if final_selection_path.exists() or final_selection_path.is_symlink():
        existing, _ = read_json_snapshot(final_selection_path)
        if existing != payload:
            raise FileExistsError(
                "기존 cross-seed final selection이 현재 두 immutable bundle과 다릅니다"
            )
        return existing
    write_json_exclusive(final_selection_path, payload)
    return payload


def _publish_selected_val_directory(
    source: Path, target: Path, *, metrics_sha256: str
) -> Path:
    """val 선택 디렉터리를 same-FS staging 뒤 no-replace로 공개한다.

    기존 target은 metrics bytes가 정확히 같은 경우에만 재진입 산출물로
    인정한다. 부분 copy나 symlink를 정상 완료로 오인하지 않는다.
    """

    source = Path(os.path.abspath(source))
    target = Path(os.path.abspath(target))
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise FileExistsError(f"canonical val 경로가 실제 directory가 아닙니다: {target}")
        current = snapshot_regular_file(target / "metrics.npz")
        if current.sha256 != metrics_sha256:
            raise FileExistsError(
                f"기존 canonical val metrics가 selection과 다릅니다: {target}"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=target.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        copied_metrics = snapshot_regular_file(staging / "metrics.npz")
        if copied_metrics.sha256 != metrics_sha256:
            raise RuntimeError("canonical val staging bytes가 selection과 다릅니다")
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_symlink():
                raise ValueError(f"canonical val staging에 symlink가 있습니다: {path}")
            if path.is_file():
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            elif path.is_dir():
                descriptor = os.open(
                    path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        descriptor = os.open(
            staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return publish_directory_noreplace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _run_cross_seed_command(
    command: list[str], *, environment: dict[str, str] | None = None
) -> None:
    child_environment = os.environ.copy()
    if environment:
        child_environment.update(environment)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=False, env=child_environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cross-seed command가 실패했습니다(exit={completed.returncode}): {command}"
        )


def run_cross_seed_campaign(args: argparse.Namespace) -> int:
    """두 val bundle을 고정한 뒤 capability→test 1회→completion을 연속 실행한다."""

    if len(args.cross_seed_selection) != 2 or not args.cross_seed_final_selection:
        raise ValueError(
            "cross-seed 모드는 --cross-seed-selection 두 개와 "
            "--cross-seed-final-selection이 필요합니다"
        )
    final_path = Path(args.cross_seed_final_selection).expanduser()
    if not final_path.is_absolute():
        final_path = REPO_ROOT / final_path
    final_path = Path(os.path.abspath(final_path))
    try:
        final_path.relative_to(Path(os.path.abspath(REPO_ROOT)))
    except ValueError as exc:
        raise ValueError("cross-seed final selection은 저장소 내부여야 합니다") from exc
    lock_path = final_path.with_suffix(final_path.suffix + ".lock")
    with ProcessLock(lock_path, role="cross-seed final test"):
        selection = finalize_cross_seed_selection(
            selection_paths=[Path(value) for value in args.cross_seed_selection],
            final_selection_path=final_path,
        )
        selected = selection["selected"]
        selected_checkpoint = Path(selected["checkpoint"])
        selected_manifest = Path(selection["manifest"])
        state = torch.load(
            io.BytesIO(snapshot_regular_file(selected_checkpoint).content),
            map_location="cpu",
            weights_only=False,
        )
        saved_cfg = state["cfg"]
        winner_seed = int(selection["seed"])
        resolved_cfg = load_train_config(
            _resolved_input_path(args.config),
            [*args.overrides, f"seed={winner_seed}"],
        )
        if (
            resolved_cfg.get("experiment_contract_sha256")
            != saved_cfg.get("experiment_contract_sha256")
        ):
            raise ValueError(
                "cross-seed completion config가 winner checkpoint contract와 다릅니다"
            )
        selected_run = selected_checkpoint.parent.parent
        canonical_val = selected_run / "eval_recorded_val"
        _publish_selected_val_directory(
            Path(selected["evaluation_dir"]),
            canonical_val,
            metrics_sha256=str(selected["metrics_sha256"]),
        )

        capability_path, consumed_path = canonical_test_ledger_paths(final_path)
        token = issue_test_capability(
            selection_path=final_path, capability_path=capability_path
        )
        test_dir = selected_run / "eval_recorded_test"
        python = sys.executable
        _run_cross_seed_command(
            [
                python,
                str(REPO_ROOT / "scripts/eval/evaluate_recorded.py"),
                "--ckpt",
                str(selected_checkpoint),
                "--manifest",
                str(selected_manifest),
                "--split",
                "test",
                "--out",
                str(test_dir),
                "--selection",
                str(final_path),
                "--test-capability",
                str(capability_path),
                "--test-consumed-marker",
                str(consumed_path),
            ],
            environment={CAPABILITY_ENV: token},
        )
        completion_out = final_path.parent / "completion"
        override_args = [
            part
            for value in [*args.overrides, f"seed={winner_seed}"]
            for part in ("--set", value)
        ]
        _run_cross_seed_command(
            [
                python,
                str(REPO_ROOT / "scripts/train/check_finetune.py"),
                "--config",
                str(_resolved_input_path(args.config)),
                *override_args,
                "--completion-checkpoint",
                str(selected_checkpoint),
                "--val-metrics",
                str(canonical_val / "metrics.npz"),
                "--test-metrics",
                str(test_dir / "metrics.npz"),
                "--selection",
                str(final_path),
                "--test-capability",
                str(capability_path),
                "--test-consumed-marker",
                str(consumed_path),
                "--out-dir",
                str(completion_out),
            ]
        )
    return EXIT_OK


def freeze_recorded_val_selection(
    *,
    candidates: list[tuple[Path, Path]],
    selection_path: Path,
    manifest_path: Path,
    experiment_contract_sha256: str,
) -> dict:
    """recorded val만 읽어 checkpoint 선택을 한 번 고정한다."""

    if selection_path.exists():
        raise FileExistsError(
            f"recorded-val 선택 산출물이 이미 있어 덮어쓸 수 없습니다: {selection_path}"
        )
    contract_sha = str(experiment_contract_sha256).lower()
    if len(contract_sha) != 64 or any(
        character not in "0123456789abcdef" for character in contract_sha
    ):
        raise ValueError("recorded-val 선택에는 64자리 experiment contract SHA가 필요합니다")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"recorded manifest가 없습니다: {manifest_path}")
    manifest_snapshot = snapshot_regular_file(manifest_path)
    manifest_sha = manifest_snapshot.sha256
    rows = []
    candidate_seeds: set[int] = set()
    campaign_shas: set[str] = set()
    for checkpoint, evaluation_dir in candidates:
        metrics_path = evaluation_dir / "metrics.npz"
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"recorded-val 선택 입력이 없습니다: {checkpoint}, {metrics_path}"
            )
        checkpoint_snapshot = snapshot_regular_file(checkpoint)
        checkpoint_sha = checkpoint_snapshot.sha256
        state = torch.load(
            io.BytesIO(checkpoint_snapshot.content),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
            raise ValueError(f"{checkpoint}: embedded resolved cfg가 없습니다")
        embedded = validate_embedded_experiment_contract(state["cfg"])
        if embedded["sha256"] != contract_sha:
            raise ValueError(
                f"{checkpoint}: 후보 embedded experiment contract가 selection과 다릅니다"
            )
        seed = state["cfg"].get("seed")
        if not isinstance(seed, int) or seed not in OFFICIAL_FINETUNE_SEEDS:
            raise ValueError(f"{checkpoint}: 공식 fine-tune seed가 아닙니다: {seed!r}")
        candidate_seeds.add(seed)
        campaign_sha = seed_neutral_campaign_sha256(state["cfg"])
        campaign_shas.add(campaign_sha)
        metrics_snapshot = snapshot_regular_file(metrics_path)
        with np.load(io.BytesIO(metrics_snapshot.content), allow_pickle=False) as data:
            required = {
                "nmse_trusted_worst10_mean_db",
                "split",
                "checkpoint_sha256",
                "manifest_sha256",
                "experiment_contract_sha256",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(
                    f"{metrics_path}: recorded-val 선택 지문이 없습니다: {missing}"
                )
            evaluated_split = str(np.asarray(data["split"]).reshape(-1)[0])
            evaluated_checkpoint_sha = str(
                np.asarray(data["checkpoint_sha256"]).reshape(-1)[0]
            )
            evaluated_manifest_sha = str(
                np.asarray(data["manifest_sha256"]).reshape(-1)[0]
            )
            evaluated_contract_sha = str(
                np.asarray(data["experiment_contract_sha256"]).reshape(-1)[0]
            )
        decision = classify_recorded_val_metrics(metrics_snapshot.content)
        metric = float(decision["selection_metric_db"])
        if evaluated_split != "val":
            raise ValueError(f"{metrics_path}: selection 입력 split이 val이 아닙니다")
        if evaluated_checkpoint_sha != checkpoint_sha:
            raise ValueError(f"{metrics_path}: checkpoint SHA가 현재 후보와 다릅니다")
        if evaluated_manifest_sha != manifest_sha:
            raise ValueError(f"{metrics_path}: manifest SHA가 현재 val과 다릅니다")
        if evaluated_contract_sha != contract_sha:
            raise ValueError(f"{metrics_path}: experiment contract SHA가 다릅니다")
        rows.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_dir": str(evaluation_dir.resolve()),
                "metrics_sha256": metrics_snapshot.sha256,
                "selection_metric": "nmse_trusted_worst10_mean_db",
                "selection_metric_db": metric,
                "seed": seed,
                "seed_neutral_campaign_sha256": campaign_sha,
                "decision": decision,
            }
        )
    if not rows:
        raise ValueError("recorded-val checkpoint 후보가 없습니다")
    if len(candidate_seeds) != 1:
        raise ValueError(f"한 run의 recorded-val 후보 seed가 서로 다릅니다: {candidate_seeds}")
    if len(campaign_shas) != 1:
        raise ValueError("한 run의 후보 seed-neutral campaign digest가 서로 다릅니다")
    clear = [row for row in rows if row["decision"]["status"] == "clear_pass"]
    borderline = [row for row in rows if row["decision"]["status"] == "borderline"]
    pool = clear or borderline or rows
    selected = max(
        pool,
        key=lambda row: (
            float(row["decision"]["minimum_margin_db"]),
            -float(row["selection_metric_db"]),
            row["checkpoint_sha256"],
        ),
    )
    payload = {
        "schema_version": 2,
        "selection_split": "val",
        "test_opened": False,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "experiment_contract_sha256": contract_sha,
        "seed_neutral_campaign_sha256": next(iter(campaign_shas)),
        "seed": next(iter(candidate_seeds)),
        "decision": selected["decision"],
        "candidates": rows,
        "selected": selected,
    }
    write_json_exclusive(selection_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cross_seed_selection or args.cross_seed_final_selection:
        try:
            return run_cross_seed_campaign(args)
        except LockHeldError as exc:
            print(f"[중복] {exc}", file=sys.stderr)
            return EXIT_PIPELINE_LOCKED
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[중단] cross-seed finalize 오류: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    # status는 학습 시작 요청이 아니다. fine-tune의 필수 measured-mode
    # override가 없어도 기존 lock/상태를 읽을 수 있어야 한다. 여기서는
    # ckpt_dir 해석에 필요한 top-level config만 읽고 학습 물리는 검증하지 않는다.
    if args.status:
        try:
            config_path = _resolved_input_path(args.config)
            status_overrides = list(args.overrides)
            # canonical guarded fine-tune run은 measured mode다. status 조회가
            # 학습 override를 반복하지 않아도 같은 contract run을 찾는다.
            if not any(
                item.split("=", 1)[0].strip() == "data.digital_primary_path_mode"
                for item in status_overrides
            ):
                status_overrides.append("data.digital_primary_path_mode=measured")
            status_cfg = load_train_config(config_path, status_overrides)
            run_dir = resolve_run_dir(status_cfg["ckpt_dir"])
            run_dir.relative_to(REPO_ROOT.resolve())
            lock_dir = autostart_state_dir(run_dir)
            state_dir = _repo_state_dir(args.state_dir) if args.state_dir else lock_dir
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            print(f"[중단] fine-tune status config 오류: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        _print_status(
            state_dir / "status.json",
            lock_dir / "pipeline.lock",
            lock_dir / "train.lock",
        )
        return EXIT_OK

    # (A) lock 이전: 설정 해석. 실패하면 어떤 파일도 만들지 않는다.
    try:
        config_path = _resolved_input_path(args.config)
        cfg = load_train_config(config_path, args.overrides)
        run_dir = resolve_run_dir(cfg["ckpt_dir"])
        run_dir.relative_to(REPO_ROOT.resolve())
        lock_dir = autostart_state_dir(run_dir)
        state_dir = _repo_state_dir(args.state_dir) if args.state_dir else lock_dir
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[중단] fine-tune config 오류: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    audit_dir = state_dir / "audit"
    status_path = state_dir / "status.json"
    pipeline_lock = lock_dir / "pipeline.lock"
    train_lock = lock_dir / "train.lock"

    # (B) --status 는 lock 없이 읽기만 한다.
    if args.status:
        _print_status(status_path, pipeline_lock, train_lock)
        return EXIT_OK

    if state_dir != lock_dir:
        print(f"[주의] --state-dir 사용: lock 은 {pipeline_lock} 로 고정됩니다", file=sys.stderr)
    _warn_legacy_run_audit(run_dir)

    # (C) readiness 감사보다 먼저 lock. 탈락자는 state_dir 에 아무것도 쓰지 않는다.
    try:
        lock = ProcessLock(
            pipeline_lock,
            role="fine-tune pipeline",
            metadata={
                "run_dir": str(run_dir),
                "state_dir": str(state_dir),
                "mode": "check-only" if args.check_only else "pipeline",
            },
        ).acquire()
    except LockHeldError as exc:
        print(f"[중복] {exc}", file=sys.stderr)
        print(
            "  진행 상황은 --status 로, 독립 감사는 check_finetune.py 로 확인하세요.",
            file=sys.stderr,
        )
        return EXIT_PIPELINE_LOCKED

    # 이미 acquire() 했으므로 ``with lock:`` 을 쓰면 __enter__ 가 같은 경로를 다시 잡으려다
    # 자기 자신과 충돌한다. 획득은 한 번, 해제는 finally 로 한다.
    try:
        run_dir_existed = run_dir.exists()
        status = PipelineStatus(
            status_path,
            mode="check-only" if args.check_only else "pipeline",
            run_key=finetune_run_key(run_dir),
            run_dir=run_dir,
            state_dir=state_dir,
            lock_path=pipeline_lock,
            config_path=config_path,
            config_sha256=sha256_text(config_path.read_text(encoding="utf-8")),
            overrides=list(args.overrides),
            fingerprint=config_fingerprint(cfg),
        )

        # 학습 단계가 있을 때만: 무거운 QA 를 돌리기 전에 중복 학습을 먼저 걸러낸다.
        if not args.check_only and _train_lock_busy(train_lock):
            print("[중단] 다른 프로세스가 같은 run 을 학습 중입니다(train.lock)", file=sys.stderr)
            return status.finish("train_locked", EXIT_TRAIN_LOCKED)

        status.update("readiness")
        readiness = audit_finetune_readiness(cfg)
        atomic_write_text(
            audit_dir / "readiness.json",
            json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(audit_dir / "readiness.md", render_audit_markdown(readiness))
        status.update(
            readiness={
                "ok": readiness["ok"],
                "report": str(audit_dir / "readiness.json"),
                "checked_at_utc": readiness.get("checked_at_utc"),
                "failed_checks": [c["id"] for c in readiness["checks"] if not c["ok"]],
            },
            # NOT READY 인데 학습 디렉터리가 생겼다면 불변식이 깨진 것이다. 배포 환경에서도
            # 회귀를 잡을 수 있게 관측값을 남긴다.
            run_dir_created_before_ready=(not run_dir_existed and run_dir.exists()),
        )
        if not readiness["ok"]:
            print("[NOT READY] fine-tune 학습을 시작하지 않습니다", file=sys.stderr)
            for item in readiness["checks"]:
                if not item["ok"]:
                    print(f"  - {item['id']}: {item['message']}", file=sys.stderr)
            return status.finish("not_ready", EXIT_NOT_READY)

        print("[READY] P/S·lead·pretrain·recorded 진입 게이트 PASS", flush=True)
        if args.check_only:
            return status.finish("ready", EXIT_OK)

        # ``.resolve()`` 를 쓰면 안 된다. venv 의 bin/python 은 시스템 인터프리터를 가리키는
        # **심볼릭 링크**라, 따라가는 순간 venv 의 site-packages 를 잃고 자식이
        # "No module named yaml" 로 죽는다. sys.executable 은 이미 절대경로다.
        python = sys.executable
        override_args = [part for value in args.overrides for part in ("--set", value)]
        ckpt_dir = run_dir / "ckpt"
        best = ckpt_dir / "best.pt"
        last = ckpt_dir / "last.pt"
        if best.exists() and not last.exists() and args.resume is None:
            print(
                f"[중단] best.pt만 있고 last.pt가 없어 안전하게 재개할 수 없습니다: {ckpt_dir}",
                file=sys.stderr,
            )
            return status.finish("failed", EXIT_CONFIG, failed_step="resume_ambiguous")

        try:
            explicit_resume = _resolve_explicit_resume(args.resume, ckpt_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[중단] {exc}", file=sys.stderr)
            return status.finish(
                "failed", EXIT_CONFIG, failed_step="resume_explicit_required"
            )

        train_command = [
            python,
            str(REPO_ROOT / "scripts" / "train" / "train.py"),
            "--config",
            str(config_path),
            *override_args,
        ]
        if explicit_resume is not None:
            train_command += ["--resume", str(explicit_resume)]

        try:
            status.update(
                "training",
                resume={
                    "resumed_from": (
                        str(explicit_resume) if explicit_resume is not None else None
                    ),
                    "detected_by": "explicit_cli",
                },
            )
            _run(train_command, step="train", status=status)
            if not best.is_file() or not last.is_file():
                raise RuntimeError("학습 종료 뒤 best.pt/last.pt가 모두 존재하지 않습니다")
            validate_completion_receipt(
                ckpt_dir,
                expected_role="canonical_finetune",
                expected_init_eligible=False,
                repo_root=REPO_ROOT,
            )

            status.update("evaluating")
            manifest_path = Path(str(cfg["recorded_manifest"]))
            if not manifest_path.is_absolute():
                manifest_path = REPO_ROOT / manifest_path

            # 합성 val best를 바로 test에 넣지 않는다. 학습이 남긴 독립
            # 후보(best/last)를 recorded val로만 비교하고 selection.json을 영구
            # 고정한 뒤에만 test를 단 한 번 개봉한다.
            candidate_pairs: list[tuple[Path, Path]] = []
            seen_checkpoint_sha: set[str] = set()
            for candidate in (best, last):
                candidate_sha = _sha256_file(candidate)
                if candidate_sha in seen_checkpoint_sha:
                    continue
                seen_checkpoint_sha.add(candidate_sha)
                candidate_out = run_dir / "eval_recorded_val_candidates" / candidate.stem
                _run(
                    [
                        python,
                        str(REPO_ROOT / "scripts" / "eval" / "evaluate_recorded.py"),
                        "--ckpt",
                        str(candidate),
                        "--manifest",
                        str(manifest_path),
                        "--split",
                        "val",
                        "--out",
                        str(candidate_out),
                    ],
                    step=f"evaluate_recorded_val_{candidate.stem}",
                    status=status,
                )
                candidate_pairs.append((candidate, candidate_out))

            selection_path = audit_dir / "recorded_val_selection.json"
            selection = freeze_recorded_val_selection(
                candidates=candidate_pairs,
                selection_path=selection_path,
                manifest_path=manifest_path,
                experiment_contract_sha256=str(
                    cfg.get("resolved_contract_run_dir", {}).get(
                        "experiment_contract_sha256", ""
                    )
                ),
            )
            val_status = str((selection.get("decision") or {}).get("status", ""))
            if int(cfg.get("seed", 0)) == 20260903:
                status.update(
                    val_selection=selection.get("decision"),
                    required_action="cross_seed_finalize",
                    test_capability_issued=False,
                )
                print(
                    "[보류] second-seed val 선택을 고정했습니다. seed 20260803 bundle과 "
                    "cross-seed finalize 전에는 test를 열지 않습니다.",
                    file=sys.stderr,
                )
                return status.finish("awaiting_cross_seed_finalize", EXIT_NOT_READY)
            if val_status == "borderline":
                status.update(
                    val_selection=selection.get("decision"),
                    required_next_seed=20260903,
                    test_capability_issued=False,
                )
                print(
                    "[보류] val G4 경계 0.3 dB 이내입니다. test를 열지 않고 "
                    "seed=20260903 실행 뒤 cross-seed finalize가 필요합니다.",
                    file=sys.stderr,
                )
                return status.finish("val_borderline", EXIT_NOT_READY)
            if val_status != "clear_pass":
                raise RuntimeError(
                    "recorded val G4 clear-pass가 아니므로 test capability를 발급하지 "
                    f"않습니다: {selection.get('decision')}"
                )
            selected = Path(selection["selected"]["checkpoint"])
            selected_val_dir = Path(selection["selected"]["evaluation_dir"])
            canonical_val_dir = run_dir / "eval_recorded_val"
            _publish_selected_val_directory(
                selected_val_dir,
                canonical_val_dir,
                metrics_sha256=str(selection["selected"]["metrics_sha256"]),
            )

            test_dir = run_dir / "eval_recorded_test"
            capability_path, consumed_marker_path = canonical_test_ledger_paths(
                selection_path
            )
            test_token = issue_test_capability(
                selection_path=selection_path,
                capability_path=capability_path,
            )
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts" / "eval" / "evaluate_recorded.py"),
                    "--ckpt",
                    str(selected),
                    "--manifest",
                    str(manifest_path),
                    "--split",
                    "test",
                    "--out",
                    str(test_dir),
                    "--selection",
                    str(selection_path),
                    "--test-capability",
                    str(capability_path),
                    "--test-consumed-marker",
                    str(consumed_marker_path),
                ],
                step="evaluate_recorded_test_once",
                status=status,
                environment={CAPABILITY_ENV: test_token},
            )

            status.update("completion")
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts" / "train" / "check_finetune.py"),
                    "--config",
                    str(config_path),
                    *override_args,
                    "--completion-checkpoint",
                    str(selected),
                    "--selection",
                    str(selection_path),
                    "--test-capability",
                    str(capability_path),
                    "--test-consumed-marker",
                    str(consumed_marker_path),
                    "--out-dir",
                    str(audit_dir),
                ],
                step="completion",
                status=status,
            )
        except StepFailed as exc:
            if exc.step == "train" and exc.returncode == EXIT_TRAIN_LOCKED:
                print(
                    "[중단] 다른 프로세스가 같은 run 을 학습 중입니다(train.lock)",
                    file=sys.stderr,
                )
                return status.finish("train_locked", EXIT_TRAIN_LOCKED)
            print(f"[FAIL] fine-tune pipeline: {exc}", file=sys.stderr)
            return status.finish("failed", EXIT_STAGE_FAILED, failed_step=exc.step)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            print(f"[FAIL] fine-tune pipeline: {exc}", file=sys.stderr)
            return status.finish("failed", EXIT_STAGE_FAILED)

        print(f"[COMPLETE] recorded-val selected + single-use test G4 PASS: {selected}")
        return status.finish(
            "done",
            EXIT_OK,
            artifacts={
                "selected": str(selected),
                "recorded_val_selection": str(selection_path),
                "recorded_test_capability": str(capability_path),
                "recorded_test_consumed": str(consumed_marker_path),
                "last": str(last),
                "eval_val": str(run_dir / "eval_recorded_val"),
                "eval_test": str(run_dir / "eval_recorded_test"),
                "completion_report": str(audit_dir / "completion.json"),
            },
        )
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
