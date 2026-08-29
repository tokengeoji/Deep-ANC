"""recorded-test completed/failed 최종 상태의 동시 전이 회귀 테스트."""

from __future__ import annotations

import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

import deep_anc.train.evaluation_contract as contract


@pytest.mark.parametrize("g4_verdict", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_completion_and_failure_cannot_both_publish(
    tmp_path: Path, monkeypatch, g4_verdict: str
) -> None:
    """두 스레드가 동시에 최종 상태를 요청해도 마커는 정확히 하나다."""

    ledger = tmp_path / "ledger"
    ledger.mkdir()
    paths = {
        "issued": ledger / "capability.json",
        "running": ledger / "consumed.json",
        "completed": ledger / "completed.json",
        "failed": ledger / "failed.json",
    }
    paths["issued"].write_text("issued\n", encoding="utf-8")
    paths["running"].write_text("running\n", encoding="utf-8")
    running_snapshot = contract.snapshot_regular_file(paths["running"])

    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_snapshot = contract.snapshot_regular_file(manifest)
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": {}}, checkpoint)
    checkpoint_sha = contract.snapshot_regular_file(checkpoint).sha256
    experiment_sha = "e" * 64
    capability_sha = "a" * 64
    selection_path = tmp_path / "selection.json"
    contract.write_json_exclusive(
        selection_path,
        {
            "manifest": str(manifest),
            "selected": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
            },
            "experiment_contract_sha256": experiment_sha,
        },
    )
    selection_snapshot = contract.snapshot_regular_file(selection_path)
    running = {
        "capability_sha256": capability_sha,
        "experiment_contract_sha256": experiment_sha,
        "seed_neutral_campaign_sha256": "d" * 64,
        "selected_checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_snapshot.sha256,
    }

    output = tmp_path / "test-output"
    output.mkdir()
    (output / "metrics.md").write_text("ok\n", encoding="utf-8")
    np.savez_compressed(
        output / "metrics.npz",
        checkpoint_sha256=np.asarray(checkpoint_sha),
        experiment_contract_sha256=np.asarray(experiment_sha),
        selection_sha256=np.asarray(selection_snapshot.sha256),
        test_capability_sha256=np.asarray(capability_sha),
        test_consumed_marker_sha256=np.asarray(running_snapshot.sha256),
    )

    def fake_active_test_ledger(**_kwargs):
        return running, running_snapshot, selection_snapshot, paths

    monkeypatch.setattr(contract, "_active_test_ledger", fake_active_test_ledger)
    monkeypatch.setattr(
        contract,
        "validate_persisted_g4_metrics",
        lambda *_args, **_kwargs: {"g4_verdict": g4_verdict},
    )

    # 두 호출이 공용 lock을 동시에 요청하도록 맞춘다. lock을 잡기 전에만 기다려야
    # 첫 스레드가 lock 안에서 두 번째 스레드를 기다리는 test deadlock이 생기지 않는다.
    lock_attempt = threading.Barrier(2)
    original_lock = contract._test_terminal_ledger_lock

    @contextmanager
    def synchronized_lock(lock_paths):
        lock_attempt.wait(timeout=5.0)
        with original_lock(lock_paths):
            yield

    monkeypatch.setattr(contract, "_test_terminal_ledger_lock", synchronized_lock)

    def complete():
        return contract.complete_test_evaluation(
            selection_path=selection_path,
            capability_path=paths["issued"],
            consumed_marker_path=paths["running"],
            output_dir=output,
            repo_root=tmp_path,
        )

    def fail():
        return contract.fail_test_evaluation(
            selection_path=selection_path,
            capability_path=paths["issued"],
            consumed_marker_path=paths["running"],
            error_type="simultaneous_failure",
            repo_root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(complete), executor.submit(fail))
        outcomes: list[Path | BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10.0))
            except BaseException as exc:  # 결과의 예외 타입도 아래에서 계약으로 검사한다.
                outcomes.append(exc)

    terminal_paths = [paths["completed"], paths["failed"]]
    published = [path for path in terminal_paths if path.exists()]
    assert len(published) == 1
    assert sum(isinstance(value, Path) for value in outcomes) == 1
    errors = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(errors) == 1
    assert isinstance(errors[0], FileExistsError)
    payload = json.loads(published[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["phase"] == published[0].stem
    if published[0] == paths["completed"]:
        assert g4_verdict == "PASS"
    elif payload.get("failure_class") == "valid_g4_terminal_rejection":
        assert g4_verdict in {"FAIL", "INCONCLUSIVE"}
        assert payload["g4_verdict"] == g4_verdict

    lock_path = ledger / ".terminal.lock"
    assert stat.S_ISREG(lock_path.lstat().st_mode)
