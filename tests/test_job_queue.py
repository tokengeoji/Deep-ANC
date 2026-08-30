"""GPU 작업 큐 감독자 회귀 테스트.

가장 중요한 회귀는 **한 작업이 실패해도 나머지가 계속 실행되는 것**이다.
기존 run_structure_search.sh 는 실패 시 exit 1 로 남은 후보를 전부 취소해 GPU 를
유휴로 만들었다. 그 결함이 되살아나면 이 파일이 먼저 깨진다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.ops.job_queue import (  # noqa: E402
    FileLock,
    LockHeldError,
    QueueSpec,
    QueueSpecError,
    StatusWriter,
    Supervisor,
    classify_failure,
    gpu_is_free,
    identity_matches,
    load_queue,
    lock_is_held,
    proc_identity,
    read_progress,
    read_status,
    wait_for_pid_exit,
)
from deep_anc.ops.structure_select import (  # noqa: E402
    bootstrap_ci,
    parse_metrics_markdown,
)


# ---------------------------------------------------------------------------
# 하네스
# ---------------------------------------------------------------------------


def _stat_line(pid: int, comm: str, starttime: str) -> str:
    """/proc/<pid>/stat 한 줄.

    ``proc_identity`` 는 마지막 ')' 뒤부터 분해하므로 tail[0]=state(field3) 이고
    starttime(field22)=tail[19] 다. 즉 state 뒤에 filler 18개를 둬야 자리가 맞는다.
    """

    return f"{pid} ({comm}) S " + " ".join(["0"] * 18) + f" {starttime} rest"


def _spec(tmp_path: Path, jobs: list[dict], **extra) -> QueueSpec:
    spec = QueueSpec(
        gpu=0,
        python=sys.executable,
        state_dir=str(tmp_path / "queue"),
        tunables={"status_interval_seconds": 0, "settle_seconds": 0, "min_free_gb": 0},
        entry_gate={"require_gpu_free": False},
        jobs=[],
        source=str(tmp_path / "queue.yaml"),
        **extra,
    )
    from deep_anc.ops.job_queue import Job

    spec.jobs = [Job(**job) for job in jobs]
    return spec


def _shell(job_id: str, exit_code: int, *, message: str = "") -> dict:
    script = f"echo '{message}'; exit {exit_code}" if message else f"exit {exit_code}"
    return {
        "id": job_id,
        "kind": "shell",
        "command": ["/bin/sh", "-c", script],
        "on_failure": "continue",
    }


def _supervisor(spec: QueueSpec, **kwargs) -> Supervisor:
    return Supervisor(
        spec, sleep=lambda _s: None, exit_when_drained=True, log=lambda _m: None, **kwargs
    )


# ---------------------------------------------------------------------------
# 실패 격리 — 이 파일의 핵심
# ---------------------------------------------------------------------------


def test_one_failing_job_does_not_cancel_the_rest(tmp_path):
    """2번째 작업이 실패해도 1·3·4 가 모두 실행되고 감독자는 정상 종료한다."""

    spec = _spec(
        tmp_path,
        [_shell("a", 0), _shell("b", 42), _shell("c", 0), _shell("d", 0)],
    )
    supervisor = _supervisor(spec)
    assert supervisor.run() == 0

    states = {job_id: record["state"] for job_id, record in supervisor.results.items()}
    assert states["a"] == "succeeded"
    assert states["b"].startswith("failed")
    # 결함이 되살아나면 c/d 가 아예 결과에 없다.
    assert states["c"] == "succeeded"
    assert states["d"] == "succeeded"


def test_transient_failure_retries_once_then_continues(tmp_path):
    job = _shell("oom", 1, message="torch.cuda.OutOfMemoryError: CUDA out of memory")
    job["on_failure"] = "retry_then_continue"
    job["max_retries"] = 1
    spec = _spec(tmp_path, [job, _shell("after", 0)])
    supervisor = _supervisor(spec)
    assert supervisor.run() == 0

    assert supervisor.results["oom"]["state"] == "failed_transient"
    assert supervisor.results["oom"]["attempt"] == 2  # 최초 + 재시도 1회
    assert supervisor.results["after"]["state"] == "succeeded"


def test_deterministic_failure_is_not_retried(tmp_path):
    job = _shell("boom", 1, message="Traceback (most recent call last)")
    job["on_failure"] = "retry_then_continue"
    job["max_retries"] = 3
    spec = _spec(tmp_path, [job])
    supervisor = _supervisor(spec)
    supervisor.run()
    assert supervisor.results["boom"]["state"] == "failed_deterministic"
    assert supervisor.results["boom"]["attempt"] == 1


def test_failed_dependency_skips_dependent_job(tmp_path):
    spec = _spec(
        tmp_path,
        [_shell("root", 7), {**_shell("child", 0), "depends_on": ["root"]}, _shell("tail", 0)],
    )
    supervisor = _supervisor(spec)
    supervisor.run()
    assert supervisor.results["child"]["state"] == "skipped_dependency"
    # 의존이 끊겨도 무관한 작업은 계속 돈다.
    assert supervisor.results["tail"]["state"] == "succeeded"


def test_restart_does_not_rerun_completed_jobs(tmp_path):
    """재기동이 완료된 작업을 다시 돌리면 시간 낭비를 넘어 **파괴적**이다.

    실제로 20k 를 완주한 대조군을 재실행해 checkpoint 를 덮어쓸 뻔했다(step 500 직전 차단).
    """

    counter = tmp_path / "runs.txt"
    job = {
        "id": "once",
        "kind": "shell",
        "command": ["/bin/sh", "-c", f"echo x >> {counter}"],
        "on_failure": "continue",
    }
    first = _spec(tmp_path, [job])
    supervisor = _supervisor(first)
    supervisor.run()
    assert counter.read_text().count("x") == 1

    # 새 감독자 인스턴스 = 재기동. 상태 파일에서 결과를 복원해야 한다.
    second = _supervisor(_spec(tmp_path, [job]))
    second.run()
    assert counter.read_text().count("x") == 1, "완료된 작업이 재실행됐다"
    assert second.results["once"]["state"] == "succeeded"


def test_completed_train_run_on_disk_is_skipped(tmp_path, monkeypatch):
    """상태 파일이 유실돼도 디스크가 완료를 증명하면 학습을 다시 시작하지 않는다."""

    spec = _spec(
        tmp_path,
        [{
            "id": "pilot", "kind": "train", "ckpt_dir": "runs/pilot",
            "expect": {"step": 20000},
        }],
    )
    supervisor = _supervisor(spec)
    monkeypatch.setattr(
        supervisor, "verify_run",
        lambda job_id, ckpt_dir, expect: {
            "id": job_id, "state": "succeeded", "detail": "검증 통과",
            "checkpoint": {"step": 20000},
        },
    )

    def _fail(*_a, **_k):
        raise AssertionError("완료된 학습을 다시 spawn 했다")

    monkeypatch.setattr(supervisor, "spawn", _fail)
    result = supervisor.run_job(spec.jobs[0])
    assert result["state"] == "already_done"


def test_already_done_satisfies_dependencies(tmp_path):
    spec = _spec(
        tmp_path,
        [{"id": "a", "kind": "shell", "command": ["true"]},
         {**_shell("b", 0), "depends_on": ["a"]}],
    )
    supervisor = _supervisor(spec)
    supervisor.results["a"] = {"id": "a", "state": "already_done", "detail": ""}
    assert supervisor.next_job().id == "b"


def test_classify_failure_reads_log_tail(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("step 1\nCUDA out of memory\n", encoding="utf-8")
    assert classify_failure(1, log) == "failed_transient"
    log.write_text("step 1\nall good\n", encoding="utf-8")
    assert classify_failure(0, log) == "succeeded"
    assert classify_failure(137, log) == "failed_killed"


# ---------------------------------------------------------------------------
# 기존 프로세스 불가침
# ---------------------------------------------------------------------------


def test_terminate_group_refuses_foreign_pgid(tmp_path):
    """감독자가 만들지 않은 프로세스 그룹에는 신호를 보낼 수 없다.

    원격에는 base 학습(PID 22554)과 구 watcher(PID 24271)가 돌고 있다.
    이 방어선이 무너지면 남의 학습을 죽일 수 있다.
    """

    supervisor = _supervisor(_spec(tmp_path, []))
    with pytest.raises(ValueError, match="소유하지 않은 pgid"):
        supervisor.terminate_group(os.getpgrp())


def test_child_env_pins_cuda_visible_devices(tmp_path, monkeypatch):
    """자식은 반드시 감독자의 GPU 하나만 봐야 한다.

    이걸 빠뜨리면 PyTorch 가 기본값 cuda:0 에 올라가 GPU1 감독자의 학습이 GPU0 의 다른
    학습 위에 겹친다. 실제로 한 번 발생시킨 회귀다.
    """

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    spec = _spec(tmp_path, [])
    spec.gpu = 1
    assert _supervisor(spec).child_env()["CUDA_VISIBLE_DEVICES"] == "1"
    spec.gpu = 0
    assert _supervisor(spec).child_env()["CUDA_VISIBLE_DEVICES"] == "0"


def test_spawned_child_actually_receives_the_pinned_device(tmp_path):
    """child_env 만 맞고 spawn 이 env 를 넘기지 않으면 의미가 없다 — 실제 자식으로 검증."""

    spec = _spec(tmp_path, [])
    spec.gpu = 1
    supervisor = _supervisor(spec)
    log = tmp_path / "env.log"
    process = supervisor.spawn(
        ["/bin/sh", "-c", 'echo "CVD=$CUDA_VISIBLE_DEVICES"'], log
    )
    process.wait(timeout=30)
    assert "CVD=1" in log.read_text(encoding="utf-8")


def test_wait_for_pid_exit_detects_pid_reuse(tmp_path):
    """cmdline 이 같아도 starttime 이 바뀌면 원래 프로세스는 죽은 것이다."""

    proc = tmp_path / "proc" / "999"
    proc.mkdir(parents=True)
    proc.joinpath("cmdline").write_bytes(b"bash\x00run_structure_search.sh\x00")
    proc.joinpath("stat").write_text(
        _stat_line(999, "bash", "111111"), encoding="utf-8"
    )

    def _rotate(_seconds):
        proc.joinpath("stat").write_text(
            _stat_line(999, "bash", "999999"),
            encoding="utf-8",
        )

    outcome = wait_for_pid_exit(
        999, ["run_structure_search.sh"], poll_seconds=0, sleep=_rotate,
        proc_root=tmp_path / "proc",
    )
    assert outcome == "identity_changed"


def test_wait_for_pid_exit_returns_absent_for_missing_pid(tmp_path):
    (tmp_path / "proc").mkdir()
    assert (
        wait_for_pid_exit(4242, ["anything"], poll_seconds=0, proc_root=tmp_path / "proc")
        == "absent"
    )


def test_wait_for_pid_exit_observes_exit(tmp_path):
    proc = tmp_path / "proc" / "555"
    proc.mkdir(parents=True)
    proc.joinpath("cmdline").write_bytes(b"python\x00train.py\x00")
    proc.joinpath("stat").write_text(
        _stat_line(555, "python", "222"), encoding="utf-8"
    )

    def _kill(_seconds):
        for child in list(proc.iterdir()):
            child.unlink()
        proc.rmdir()

    assert (
        wait_for_pid_exit(555, ["train.py"], poll_seconds=0, sleep=_kill,
                          proc_root=tmp_path / "proc")
        == "exited"
    )


def test_proc_identity_handles_comm_with_spaces(tmp_path):
    """/proc/<pid>/stat 의 comm 필드는 괄호 안에 공백을 포함할 수 있다."""

    proc = tmp_path / "proc" / "77"
    proc.mkdir(parents=True)
    proc.joinpath("cmdline").write_bytes(b"my prog\x00--flag\x00")
    proc.joinpath("stat").write_text(
        _stat_line(77, "my prog x", "4242"), encoding="utf-8"
    )
    identity = proc_identity(77, proc_root=tmp_path / "proc")
    assert identity is not None
    assert identity["starttime"] == "4242"
    assert identity_matches(identity, ["--flag"])
    assert not identity_matches(identity, ["absent"])


def test_gpu_is_free_refuses_when_nvidia_smi_unavailable(monkeypatch):
    """조회 실패를 '유휴'로 해석하면 남의 학습 위에 작업을 얹게 된다."""

    monkeypatch.setattr("deep_anc.ops.job_queue._run_text", lambda *a, **k: "")
    free, reason = gpu_is_free(0)
    assert free is False
    assert "조회 실패" in reason


def test_gpu_is_free_refuses_when_memory_is_high(monkeypatch):
    def _fake(command, **_kwargs):
        return "" if "--query-compute-apps=pid" in " ".join(command) else "68, 10034\n"

    monkeypatch.setattr("deep_anc.ops.job_queue._run_text", _fake)
    free, reason = gpu_is_free(0, memory_threshold_mib=1024)
    assert free is False
    assert "10034" in reason


def test_gpu_is_free_accepts_idle_gpu(monkeypatch, tmp_path):
    def _fake(command, **_kwargs):
        return "" if "--query-compute-apps=pid" in " ".join(command) else "0, 12\n"

    monkeypatch.setattr("deep_anc.ops.job_queue._run_text", _fake)
    (tmp_path / "proc").mkdir()
    free, _reason = gpu_is_free(0, proc_root=tmp_path / "proc")
    assert free is True


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------


def test_second_lock_acquire_raises_and_reports_owner(tmp_path):
    path = tmp_path / "queue.lock"
    with FileLock(path, role="test", metadata={"note": "first"}):
        assert lock_is_held(path)
        with pytest.raises(LockHeldError, match="owner="):
            FileLock(path, role="test").acquire()
    assert not lock_is_held(path)


def test_lock_is_reacquirable_after_release_and_truncates(tmp_path):
    path = tmp_path / "queue.lock"
    FileLock(path, role="first").acquire().release()
    FileLock(path, role="second").acquire().release()
    # "a+" 모드로 열기 때문에 truncate 를 빠뜨리면 두 레코드가 누적된다.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["role"] == "second"


def test_stale_lock_file_is_reusable(tmp_path):
    """커널이 종료 시 해제하므로 남은 lock 파일은 그대로 재사용 가능해야 한다."""

    path = tmp_path / "queue.lock"
    path.write_text('{"role": "dead", "pid": 999999}\n', encoding="utf-8")
    assert not lock_is_held(path)
    FileLock(path, role="fresh").acquire().release()


def test_supervisor_own_lock_blocks_second_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr("deep_anc.ops.job_queue.REPO_ROOT", tmp_path)
    first = _supervisor(_spec(tmp_path, []))
    first.acquire_own_lock()
    try:
        with pytest.raises(LockHeldError):
            _supervisor(_spec(tmp_path, [])).acquire_own_lock()
    finally:
        first.close()


# ---------------------------------------------------------------------------
# 큐 스키마
# ---------------------------------------------------------------------------


def _write_queue(tmp_path: Path, payload: dict) -> Path:
    import yaml

    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_queue_rejects_unknown_key_and_duplicate_id(tmp_path):
    with pytest.raises(QueueSpecError, match="알 수 없는 키"):
        load_queue(_write_queue(tmp_path, {"gpu": 1, "jobs": [{"id": "a", "bogus": 1}]}))
    with pytest.raises(QueueSpecError, match="중복"):
        load_queue(
            _write_queue(
                tmp_path,
                {"gpu": 1, "jobs": [
                    {"id": "a", "kind": "shell", "command": ["true"]},
                    {"id": "a", "kind": "shell", "command": ["true"]},
                ]},
            )
        )


def test_queue_rejects_dangling_dependency(tmp_path):
    with pytest.raises(QueueSpecError, match="depends_on"):
        load_queue(
            _write_queue(
                tmp_path,
                {"gpu": 1, "jobs": [
                    {"id": "a", "kind": "shell", "command": ["true"], "depends_on": ["ghost"]}
                ]},
            )
        )


def test_queue_enforces_comparability_between_pilots(tmp_path):
    """후보끼리 batch/스케줄이 다르면 20k 결과를 나란히 놓을 수 없다."""

    payload = {
        "gpu": 1,
        "jobs": [
            {"id": "a", "kind": "train", "ckpt_dir": "runs/a", "compare": True,
             "overrides": {"batch_size": 128, "schedule.total_steps": 100000}},
            {"id": "b", "kind": "train", "ckpt_dir": "runs/b", "compare": True,
             "overrides": {"batch_size": 96, "schedule.total_steps": 100000}},
        ],
    }
    with pytest.raises(QueueSpecError, match="batch_size"):
        load_queue(_write_queue(tmp_path, payload))


def test_real_queue_files_load(tmp_path):
    for name in ("queue_gpu0.yaml", "queue_gpu1.yaml"):
        spec = load_queue(REPO_ROOT / "configs" / "elice" / name)
        assert spec.jobs
        assert spec.gpu in (0, 1)
        assert spec.execution_class == "legacy_diagnostic"


def test_real_legacy_queue_cannot_start_without_explicit_acknowledgement():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/elice/job_queue.py",
            "run",
            "--queue",
            "configs/elice/queue_gpu1.yaml",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--allow-legacy-diagnostic" in result.stderr


def test_real_queues_do_not_share_job_ids_or_ckpt_dirs():
    """같은 run 이 두 큐에 있으면 두 감독자가 checkpoint 를 서로 덮어쓴다.

    실제로 GPU0 이 GPU1 의 완료된 20k run 을 step 1000 으로 덮어썼다.
    """

    seen: dict[tuple[str, str], str] = {}
    for name in ("queue_gpu0.yaml", "queue_gpu1.yaml"):
        for job in load_queue(REPO_ROOT / "configs" / "elice" / name).jobs:
            for kind, value in (("id", job.id), ("ckpt_dir", job.ckpt_dir)):
                if value is None:
                    continue
                key = (kind, value)
                assert key not in seen, f"{kind}={value} 가 {seen[key]} 와 {name} 에 중복"
                seen[key] = name


def test_ckpt_dir_guard_sees_other_gpus_processes(tmp_path):
    """ckpt_dir 점유 검사는 GPU 를 가리지 않아야 한다.

    자기 GPU 프로세스만 보면 다른 GPU 감독자가 같은 run 을 돌리는 것을 놓친다.
    """

    from deep_anc.ops.job_queue import processes_with_ckpt_dir

    proc = tmp_path / "proc" / "4321"
    proc.mkdir(parents=True)
    proc.joinpath("cmdline").write_bytes(
        b"python\x00scripts/train/train.py\x00--set\x00ckpt_dir=runs/shared\x00"
    )
    proc.joinpath("stat").write_text(_stat_line(4321, "python", "9"), encoding="utf-8")
    # 이 프로세스에는 CUDA_VISIBLE_DEVICES=1 이 붙어 있지만, GPU0 감독자도 봐야 한다.
    proc.joinpath("environ").write_bytes(b"CUDA_VISIBLE_DEVICES=1\x00")

    found = processes_with_ckpt_dir("runs/shared", proc_root=tmp_path / "proc")
    assert [p["pid"] for p in found] == [4321]
    assert processes_with_ckpt_dir("runs/other", proc_root=tmp_path / "proc") == []


def test_train_command_uses_set_overrides(tmp_path):
    spec = _spec(
        tmp_path,
        [{
            "id": "t", "kind": "train", "ckpt_dir": "runs/t",
            "model_config": "configs/model_tiny.yaml",
            "overrides": {"batch_size": 128, "run_until_step": 20000},
        }],
    )
    command = _supervisor(spec).train_command(spec.jobs[0])
    # train.py 의 CLI 는 --config/--resume/--set 3개뿐이다.
    assert "--set" in command
    assert "batch_size=128" in command
    assert "run_until_step=20000" in command
    assert "ckpt_dir=runs/t" in command
    assert "model_config=configs/model_tiny.yaml" in command


# ---------------------------------------------------------------------------
# 상태 기록 / 진행도
# ---------------------------------------------------------------------------


def test_status_write_is_atomic_and_readable(tmp_path):
    writer = StatusWriter(tmp_path / "gpu1.json", tmp_path / "events.jsonl")
    writer.set(state="running", idle_seconds_total=0.0)
    writer.event("job_result", id="a", state="succeeded")
    payload = read_status(tmp_path / "gpu1.json")
    assert payload["state"] == "running"
    assert payload["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())["event"] == "job_result"


def test_read_progress_parses_training_log(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "step   12700 | loss  -14.855 | nmse_t  -15.36 dB | nmse_f  -14.62 dB | lr 9.68e-04 |  3.55 it/s\n"
        "step   12800 | loss  -13.320 | nmse_t  -14.17 dB | nmse_f  -13.30 dB | lr 9.67e-04 |  3.53 it/s\n",
        encoding="utf-8",
    )
    progress = read_progress(log)
    assert progress["step"] == 12800
    assert progress["rate_it_s"] == pytest.approx(3.53)
    assert progress["finished"] is False

    log.write_text(
        "step   20000 | loss -1 | nmse_t -1 dB | nmse_f -1 dB | lr 1e-4 |  3.5 it/s\n"
        "학습 구간 종료: step 20000/100000, best trusted val NMSE -16.67 dB\n",
        encoding="utf-8",
    )
    progress = read_progress(log)
    assert progress["finished"] is True
    assert progress["finished_step"] == 20000


# ---------------------------------------------------------------------------
# 승자 선정 보조
# ---------------------------------------------------------------------------


def test_metrics_markdown_parser_separates_band_and_source_tables(tmp_path):
    """밴드 표와 소스 표는 행 모양이 겹쳐서 섹션으로만 구분할 수 있다.

    밴드 열은 **감쇠**(양수가 좋다), 소스 열은 **NMSE**(음수가 좋다)로 부호가 반대다.
    """

    markdown = tmp_path / "metrics.md"
    markdown.write_text(
        "# 오프라인 평가\n\n"
        "## 기능1 — 주파수 대역별 감쇠 (저주파+고주파)\n\n"
        "| 밴드(Hz) | 감쇠(dB) | 신뢰 |\n|---|---|---|\n"
        "| 125 | +7.55 | 낮음* |\n| 250 | +8.46 | O |\n| 500 | -1.20 | O |\n\n"
        "## 기능2 — 소스 종류별 감쇠 (모든 소리 제거)\n\n"
        "| 소스 | NMSE(dB) | 감쇠(dB) |\n|---|---|---|\n"
        "| speech | -25.42 | +25.42 |\n| demand | -8.74 | +8.74 |\n",
        encoding="utf-8",
    )
    parsed = parse_metrics_markdown(markdown)
    assert parsed["per_source_db"] == {"speech": -25.42, "demand": -8.74}
    assert len(parsed["bands"]) == 3
    trusted = [b for b in parsed["bands"] if b["trusted"]]
    assert {b["center_hz"] for b in trusted} == {250, 500}
    assert sum(1 for b in trusted if b["attenuation_db"] <= 0) == 1


def test_metrics_markdown_parser_matches_real_output():
    """실제 run 산출물로 파서를 고정한다(포맷이 바뀌면 여기서 깨진다)."""

    real = REPO_ROOT / "runs" / "search_tiny_long" / "eval_pilot_best" / "metrics.md"
    if not real.exists():
        pytest.skip("실측 metrics.md 없음")
    parsed = parse_metrics_markdown(real)
    assert "speech" in parsed["per_source_db"]
    assert "125" not in parsed["per_source_db"]  # 밴드 행이 소스로 새면 안 된다
    assert len(parsed["bands"]) == 7


def test_config_fingerprint_ignores_architecture_but_catches_drift(tmp_path):
    """구조는 실험의 독립변수다 — 지문이 이걸로 갈리면 구조 탐색이 불가능해진다.

    실제로 첫 승자 선정에서 후보 3종이 전부 이 오탐으로 실격됐다.
    """

    import yaml

    from deep_anc.ops.structure_select import config_fingerprint

    base = {
        "batch_size": 128,
        "schedule": {"total_steps": 100000, "warmup_steps": 1250},
        "data": {"reference_mode": "digital"},
        "loss": {"nmse_objective": "trusted_band"},
    }

    def _write(name, extra):
        run = tmp_path / name
        run.mkdir()
        (run / "config_snapshot.yaml").write_text(
            yaml.safe_dump({**base, **extra}, allow_unicode=True), encoding="utf-8"
        )
        return run

    control = _write("control", {
        "ckpt_dir": "runs/search_tiny_control",
        "model_config": "configs/model_tiny.yaml",
        "model": {"name": "hybrid_anc_tiny"},
        "seed": 20260802,
    })
    variant = _write("variant", {
        "ckpt_dir": "runs/search_tiny_attn",
        "model_config": "configs/model_tiny_attn.yaml",
        "model": {"name": "hybrid_anc_tiny_attn"},
        "seed": 20260802,
    })
    drifted = _write("drifted", {
        "ckpt_dir": "runs/search_other",
        "model_config": "configs/model_tiny.yaml",
        "model": {"name": "hybrid_anc_tiny"},
        "batch_size": 96,  # 비교 가능성을 깨는 진짜 드리프트
    })

    assert config_fingerprint(control) == config_fingerprint(variant)
    assert config_fingerprint(control) != config_fingerprint(drifted)


def test_bootstrap_ci_is_deterministic_and_detects_shift():
    deltas = [-1.0] * 32
    mean, low, high = bootstrap_ci(deltas)
    assert mean == pytest.approx(-1.0)
    assert high < -0.30
    assert bootstrap_ci(deltas) == (mean, low, high)

    noisy = [0.5, -0.5] * 16
    _mean, _low, noisy_high = bootstrap_ci(noisy)
    assert noisy_high > -0.30  # 유의하지 않다 → 승자는 대조군 유지
