"""GPU 작업 큐 감독자 — GPU 가 유휴로 빠지지 않게 작업을 이어 붙인다.

왜 필요한가
-----------
저장소의 기존 실행 스크립트에는 "작업 완료 → 다음 작업 투입" 체인이 없다.
``scripts/elice/run_structure_search.sh`` 는 마지막 후보 뒤 echo 한 줄 남기고 exit 하고,
``Trainer`` 는 목표 step 도달 시 그냥 종료한다. 그래서 학습이 끝나는 순간부터
사람이 개입할 때까지 GPU 가 논다. Elice 는 인스턴스 가동 시간으로 과금되므로 이 유휴가
곧 비용이다.

더 나쁜 것은 기존 watcher 가 **작업 하나가 실패하면 ``exit 1`` 로 남은 작업을 전부
취소**한다는 점이다(run_structure_search.sh:381-413). 감독자는 그 반대로 동작한다 —
어떤 실패에도 종료하지 않고 다음 작업으로 넘어간다. 종료하는 순간 GPU 가 놀기 때문이다.

기존 프로세스 불가침
-------------------
이 감독자는 이미 돌고 있는 학습/watcher 를 절대 방해하지 않는다. 진입은 4중 AND 이며
(자기 중복 방지 / 점유자 종료 확인 / 기존 lock 획득 / GPU 실제 유휴 3회 연속),
신호는 자신이 ``start_new_session=True`` 로 직접 만든 프로세스 그룹에만 보낼 수 있다.

의존
----
표준 라이브러리 + PyYAML 만 쓴다. ``torch`` 는 checkpoint 검증에서만 지연 import 한다
(진행도 추적은 100MB 급 checkpoint 대신 학습 로그 tail 로 읽는다).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA_VERSION = 1

# 학습 로그 진행 줄: "step   12800 | loss ... |  3.53 it/s"
_STEP_LINE = re.compile(r"^step\s+(\d+)\b.*?([\d.]+)\s*it/s\s*$")
# 정상 종료 줄: "학습 구간 종료: step 20000/100000, best trusted val NMSE -16.67 dB"
_FINISH_LINE = re.compile(r"학습 구간 종료:\s*step\s+(\d+)/(\d+)")

_TRANSIENT_PATTERNS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "CUDA error",
    "NCCL",
    "cuDNN error",
)
_DETERMINISTIC_PATTERNS = (
    "Traceback (most recent call last)",
    "ValueError",
    "KeyError",
    "FileNotFoundError",
    "ModuleNotFoundError",
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def kst_now_str() -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _abs(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------


class LockHeldError(RuntimeError):
    """다른 프로세스가 같은 lock 을 보유하고 있다."""


class FileLock:
    """flock 기반 non-blocking 배타 lock + owner 메타데이터.

    ``deep_anc.train.process_lock.ProcessLock`` 과 동일한 규약이지만 의도적으로
    import 하지 않고 재구현했다 — 원격 워크트리 스냅샷에 그 모듈이 없을 수 있다.
    동등성은 ``tests/test_job_queue_lock.py`` 가 강제한다.

    파일의 존재가 아니라 실제 advisory lock 보유 여부만 권한으로 쓴다. 커널이
    프로세스 종료 시 자동 해제하므로 stale lock 파일은 그대로 재사용 가능하다.
    """

    def __init__(self, path: str | Path, *, role: str, metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.role = str(role)
        self.metadata = dict(metadata or {})
        self._handle: IO[str] | None = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            message = f"이미 실행 중인 {self.role} lock: {self.path}"
            if owner:
                message += f"; owner={owner}"
            raise LockHeldError(message) from exc

        record: dict[str, Any] = {
            "schema_version": 1,
            "role": self.role,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": utc_now_iso(),
            **self.metadata,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(record, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, _t, _v, _tb) -> None:
        self.release()


def lock_owner(path: str | Path) -> dict | None:
    """lock 파일의 owner 레코드를 읽는다(정보용 — 죽은 PID 일 수 있다)."""

    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def lock_is_held(path: str | Path) -> bool:
    """실제 advisory lock 보유 여부. 파일 존재가 아니라 flock 으로 판정한다."""

    target = Path(path)
    if not target.exists():
        return False
    try:
        handle = target.open("a+", encoding="utf-8")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# 프로세스 정체성
# ---------------------------------------------------------------------------


def proc_identity(pid: int, *, proc_root: str | Path = "/proc") -> dict | None:
    """살아 있는 프로세스의 (cmdline, starttime) 정체성. 없으면 None.

    ``starttime``(/proc/<pid>/stat 22번 필드)까지 보는 이유: PID 는 재사용된다.
    cmdline 만 검사하면 같은 PID 를 다른 프로세스가 물려받았을 때 영원히 기다린다
    (기존 ``tiny_pid_is_expected`` 가 최초 1회만 검사해 가진 함정).
    """

    base = Path(proc_root) / str(int(pid))
    try:
        cmdline = (base / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", "replace"
        ).strip()
        stat = (base / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # comm 필드는 괄호 안에 공백을 포함할 수 있으므로 마지막 ')' 뒤부터 분해한다.
    tail = stat[stat.rfind(")") + 1 :].split()
    starttime = tail[19] if len(tail) > 19 else ""
    return {"pid": int(pid), "cmdline": cmdline, "starttime": starttime}


def identity_matches(identity: dict | None, patterns: Sequence[str]) -> bool:
    if identity is None:
        return False
    cmdline = identity.get("cmdline", "")
    return all(str(p) in cmdline for p in patterns)


def wait_for_pid_exit(
    pid: int,
    patterns: Sequence[str],
    *,
    poll_seconds: float,
    on_poll: Callable[[dict], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    proc_root: str | Path = "/proc",
    max_polls: int | None = None,
) -> str:
    """점유 프로세스가 사라질 때까지 대기한다.

    반환값: ``"absent"``(처음부터 없음) / ``"exited"``(종료 관측) /
    ``"identity_changed"``(PID 재사용 — 원래 프로세스는 이미 죽음).
    셋 다 "점유자 없음"으로 취급해도 안전하다.
    """

    identity = proc_identity(pid, proc_root=proc_root)
    if identity is None:
        return "absent"
    if not identity_matches(identity, patterns):
        return "identity_changed"
    baseline = identity.get("starttime")

    polls = 0
    while True:
        if on_poll is not None:
            on_poll(identity)
        sleep(poll_seconds)
        polls += 1
        current = proc_identity(pid, proc_root=proc_root)
        if current is None:
            return "exited"
        if current.get("starttime") != baseline or not identity_matches(current, patterns):
            return "identity_changed"
        identity = current
        if max_polls is not None and polls >= max_polls:
            return "still_running"


def _run_text(command: Sequence[str], *, timeout: float = 20.0) -> str:
    try:
        done = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def processes_using_gpu(
    index: int,
    *,
    patterns: Sequence[str] = ("train.py", "evaluate_offline.py", "export_onnx.py"),
    proc_root: str | Path = "/proc",
) -> list[dict]:
    """``CUDA_VISIBLE_DEVICES`` 가 이 GPU 를 가리키는 관심 프로세스 목록.

    컨테이너에서 ``nvidia-smi --query-compute-apps`` 가 비어 보일 수 있어서
    (``--pid=host`` 없이는 목록이 비거나 권한 부족) 이 독립 신호가 필요하다.
    """

    found: list[dict] = []
    root = Path(proc_root)
    try:
        entries = [e for e in root.iterdir() if e.name.isdigit()]
    except OSError:
        return found
    for entry in entries:
        identity = proc_identity(int(entry.name), proc_root=proc_root)
        if identity is None:
            continue
        cmdline = identity["cmdline"]
        if not any(str(p) in cmdline for p in patterns):
            continue
        try:
            environ = (entry / "environ").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        visible = ""
        for item in environ.split("\x00"):
            if item.startswith("CUDA_VISIBLE_DEVICES="):
                visible = item.split("=", 1)[1]
                break
        if visible.strip() == str(index):
            found.append({**identity, "cuda_visible_devices": visible})
    return found


def processes_with_ckpt_dir(
    ckpt_dir: str, *, proc_root: str | Path = "/proc"
) -> list[dict]:
    """``ckpt_dir=<경로>`` 를 cmdline 에 가진 학습 프로세스 — **GPU 를 가리지 않는다.**

    두 GPU 감독자가 같은 run 을 돌리면 checkpoint 를 서로 덮어쓴다. 자기 GPU 의
    프로세스만 보는 검사로는 이걸 잡을 수 없다(실제로 놓쳤다).
    """

    found: list[dict] = []
    root = Path(proc_root)
    needle = f"ckpt_dir={ckpt_dir}"
    try:
        entries = [e for e in root.iterdir() if e.name.isdigit()]
    except OSError:
        return found
    for entry in entries:
        identity = proc_identity(int(entry.name), proc_root=proc_root)
        if identity is None:
            continue
        cmdline = identity["cmdline"]
        if "train.py" in cmdline and needle in cmdline:
            found.append(identity)
    return found


def gpu_snapshot(index: int) -> dict:
    """nvidia-smi 로 본 GPU 상태. 조회 실패는 ``available=False`` 로 표시한다."""

    line = _run_text(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    apps = _run_text(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ]
    ).strip()
    snapshot: dict[str, Any] = {
        "index": index,
        "available": bool(line),
        "utilization_pct": None,
        "memory_used_mib": None,
        "compute_pids": [p.strip() for p in apps.splitlines() if p.strip()],
    }
    if line:
        parts = [p.strip() for p in line.split(",")]
        try:
            snapshot["utilization_pct"] = int(float(parts[0]))
            snapshot["memory_used_mib"] = int(float(parts[1]))
        except (IndexError, ValueError):
            pass
    return snapshot


def gpu_is_free(
    index: int,
    *,
    memory_threshold_mib: int = 1024,
    patterns: Sequence[str] = ("train.py", "evaluate_offline.py", "export_onnx.py"),
    proc_root: str | Path = "/proc",
) -> tuple[bool, str]:
    """GPU 가 실제로 비었는지 3중 AND 로 판정한다.

    (a) compute-apps 목록이 비었고 (b) memory.used 가 임계 미만이며
    (c) 이 GPU 를 가리키는 관심 프로세스가 0개. 컨테이너에서 (a) 가 공허하게
    통과할 수 있으므로 (b)/(c) 가 실질적 방어선이다.
    """

    snapshot = gpu_snapshot(index)
    if snapshot["compute_pids"]:
        return False, f"compute-apps 점유: {snapshot['compute_pids']}"
    memory = snapshot["memory_used_mib"]
    if memory is None:
        return False, "nvidia-smi 조회 실패 — 유휴로 단정하지 않는다"
    if memory >= memory_threshold_mib:
        return False, f"memory.used {memory}MiB >= {memory_threshold_mib}MiB"
    foreign = processes_using_gpu(index, patterns=patterns, proc_root=proc_root)
    if foreign:
        return False, "CUDA_VISIBLE_DEVICES 점유: " + ", ".join(
            f"pid {p['pid']}" for p in foreign
        )
    return True, f"free (memory.used {memory}MiB)"


# ---------------------------------------------------------------------------
# checkpoint / 로그 판독
# ---------------------------------------------------------------------------


def checkpoint_info(path: str | Path) -> dict | None:
    """checkpoint 의 완료 판정용 필드를 읽는다. torch 는 여기서만 지연 import."""

    target = _abs(path)
    if not target.exists():
        return None
    try:
        import torch  # noqa: PLC0415 — 지연 import 가 의도다
    except ImportError:
        return None
    try:
        state = torch.load(target, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — 손상 checkpoint 도 "판정 불가"로 다룬다
        return None
    cfg = state.get("cfg") or {}
    schedule = cfg.get("schedule") or {}
    return {
        "path": str(target),
        "step": int(state.get("step", -1)),
        "best_metric": state.get("best_metric"),
        "model_name": (cfg.get("model") or {}).get("name"),
        "physics_status": cfg.get("physics_status"),
        "lead": cfg.get("digital_reference_lead_samples"),
        "total_steps": schedule.get("total_steps"),
        "run_until_step": cfg.get("run_until_step"),
        "seed": cfg.get("seed"),
        "batch_size": cfg.get("batch_size"),
    }


def tail_lines(path: str | Path, count: int = 400) -> list[str]:
    target = _abs(path)
    if not target.exists():
        return []
    try:
        with target.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = min(size, max(4096, count * 200))
            handle.seek(size - block)
            data = handle.read(block)
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()[-count:]


def read_progress(log_path: str | Path) -> dict:
    """학습 로그 tail 에서 step/속도/종료 여부를 읽는다.

    checkpoint(base 는 100MB 급)를 폴링마다 torch.load 하지 않기 위한 경로다.
    """

    progress: dict[str, Any] = {"step": None, "rate_it_s": None, "finished": False}
    for line in reversed(tail_lines(log_path, 200)):
        stripped = line.strip()
        if progress["step"] is None:
            match = _STEP_LINE.match(stripped)
            if match:
                progress["step"] = int(match.group(1))
                progress["rate_it_s"] = float(match.group(2))
        if not progress["finished"]:
            done = _FINISH_LINE.search(stripped)
            if done:
                progress["finished"] = True
                progress["finished_step"] = int(done.group(1))
        if progress["step"] is not None and progress["finished"]:
            break
    return progress


def classify_failure(returncode: int, log_path: str | Path) -> str:
    """종료 코드 + 로그 tail 로 실패 유형을 분류한다.

    재시도 정책이 여기서 갈린다. OOM/CUDA 오류는 1회 재시도하되 **batch_size 를
    자동으로 낮추지 않는다** — 하이퍼파라미터가 달라지면 후보 비교가 무효가 된다.
    """

    if returncode == 0:
        return "succeeded"
    text = "\n".join(tail_lines(log_path, 300))
    if any(p in text for p in _TRANSIENT_PATTERNS):
        return "failed_transient"
    if any(p in text for p in _DETERMINISTIC_PATTERNS):
        return "failed_deterministic"
    if returncode in (137, 143):
        return "failed_killed"
    return "failed_deterministic"


# ---------------------------------------------------------------------------
# 큐 스키마
# ---------------------------------------------------------------------------


class QueueSpecError(ValueError):
    """큐 정의가 잘못됐다."""


_JOB_KINDS = {"train", "eval", "decision", "shell", "bundle"}
_ON_FAILURE = {"continue", "retry_then_continue"}

# 후보 비교의 공정성을 지키는 오버라이드 — 큐 안의 모든 pilot 이 같아야 한다.
_COMPARABILITY_KEYS = (
    "batch_size",
    "num_workers",
    "schedule.total_steps",
    "schedule.warmup_steps",
    "eval_every",
)


@dataclass
class Job:
    id: str
    kind: str = "train"
    tier: str = "A"
    reason: str = ""
    model_config: str | None = None
    ckpt_dir: str | None = None
    resume: str | None = None
    config: str = "configs/train_pretrain.yaml"
    overrides: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)
    post_eval: list = field(default_factory=list)
    copy_before_start: list = field(default_factory=list)
    command: list = field(default_factory=list)
    on_failure: str = "retry_then_continue"
    max_retries: int = 1
    long: bool = False
    depends_on: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    compare: bool = False


@dataclass
class QueueSpec:
    gpu: int
    python: str = ".venv/bin/python"
    state_dir: str = "runs/queue"
    tunables: dict = field(default_factory=dict)
    entry_gate: dict = field(default_factory=dict)
    adopt: list = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    execution_class: str = "canonical"
    source: str = ""

    def tunable(self, name: str, default: Any) -> Any:
        env = os.environ.get("JOBQ_" + name.upper())
        if env is not None:
            try:
                return type(default)(env)
            except (TypeError, ValueError):
                pass
        value = self.tunables.get(name, default)
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default


def load_queue(path: str | Path) -> QueueSpec:
    target = _abs(path)
    if not target.exists():
        raise QueueSpecError(f"큐 정의 파일이 없습니다: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise QueueSpecError(f"큐 정의는 매핑이어야 합니다: {target}")
    version = int(raw.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise QueueSpecError(f"지원하지 않는 schema_version: {version}")
    if "gpu" not in raw:
        raise QueueSpecError("큐 정의에 gpu 가 없습니다")

    jobs: list[Job] = []
    seen: set[str] = set()
    known = set(Job.__dataclass_fields__)
    for entry in raw.get("jobs") or []:
        if not isinstance(entry, dict) or "id" not in entry:
            raise QueueSpecError(f"job 항목에 id 가 없습니다: {entry!r}")
        unknown = set(entry) - known
        if unknown:
            raise QueueSpecError(f"job {entry['id']}: 알 수 없는 키 {sorted(unknown)}")
        job = Job(**entry)
        if job.id in seen:
            raise QueueSpecError(f"job id 중복: {job.id}")
        if job.kind not in _JOB_KINDS:
            raise QueueSpecError(f"job {job.id}: 알 수 없는 kind {job.kind}")
        if job.on_failure not in _ON_FAILURE:
            raise QueueSpecError(f"job {job.id}: 알 수 없는 on_failure {job.on_failure}")
        if job.kind == "train" and not job.ckpt_dir:
            raise QueueSpecError(f"job {job.id}: train 은 ckpt_dir 이 필요합니다")
        if job.kind == "shell" and not job.command:
            raise QueueSpecError(f"job {job.id}: shell 은 command 가 필요합니다")
        seen.add(job.id)
        jobs.append(job)

    adopt_ids = {a.get("id") for a in raw.get("adopt") or [] if isinstance(a, dict)}
    for job in jobs:
        for dependency in job.depends_on:
            if dependency not in seen and dependency not in adopt_ids:
                raise QueueSpecError(f"job {job.id}: 존재하지 않는 depends_on {dependency}")

    spec = QueueSpec(
        gpu=int(raw["gpu"]),
        python=str(raw.get("python", ".venv/bin/python")),
        state_dir=str(raw.get("state_dir", "runs/queue")),
        tunables=dict(raw.get("tunables") or {}),
        entry_gate=dict(raw.get("entry_gate") or {}),
        adopt=list(raw.get("adopt") or []),
        jobs=jobs,
        execution_class=str(raw.get("execution_class", "canonical")),
        source=str(target),
    )
    if spec.execution_class not in {"canonical", "legacy_diagnostic"}:
        raise QueueSpecError(
            "execution_class는 canonical 또는 legacy_diagnostic이어야 합니다: "
            f"{spec.execution_class!r}"
        )
    validate_comparability(spec)
    return spec


def validate_comparability(spec: QueueSpec) -> None:
    """``compare: true`` 인 학습 작업들의 비교 가능성 오버라이드가 동일한지 검사한다.

    후보들이 서로 다른 batch/워커/스케줄로 돌면 20k 결과를 나란히 놓을 수 없다.
    실수로 하나만 바꾸는 사고를 스키마 단계에서 막는다.
    """

    pilots = [j for j in spec.jobs if j.kind == "train" and j.compare]
    if len(pilots) < 2:
        return
    reference = pilots[0]
    for job in pilots[1:]:
        for key in _COMPARABILITY_KEYS:
            left = reference.overrides.get(key)
            right = job.overrides.get(key)
            if left != right:
                raise QueueSpecError(
                    f"비교 대상 {reference.id} 와 {job.id} 의 {key} 가 다릅니다: "
                    f"{left!r} != {right!r}"
                )


# ---------------------------------------------------------------------------
# 상태 기록
# ---------------------------------------------------------------------------


def atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)


def read_status(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class StatusWriter:
    """감독자 상태를 원자적으로 기록한다(독자가 부분 JSON 을 보지 않도록)."""

    def __init__(self, path: str | Path, events_path: str | Path) -> None:
        self.path = Path(path)
        self.events_path = Path(events_path)
        self.data: dict[str, Any] = {}

    def set(self, **fields: Any) -> None:
        self.data.update(fields)
        self.flush()

    def flush(self) -> None:
        self.data["generated_at_utc"] = utc_now_iso()
        self.data["generated_at_kst"] = kst_now_str()
        self.data["schema_version"] = SCHEMA_VERSION
        atomic_write_text(
            self.path, json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )

    def event(self, event_name: str, /, **fields: Any) -> None:
        # 위치 전용 인자여야 한다 — 작업 레코드 자체가 "kind"/"name" 키를 갖고 있어서
        # 키워드로 받으면 ``event("job_result", kind="train")`` 이 인자 충돌을 낸다.
        record = {"at_utc": utc_now_iso(), "event": event_name, **fields}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# 감독자
# ---------------------------------------------------------------------------


class Supervisor:
    """GPU 하나를 소유하고 큐를 순차 실행한다.

    핵심 불변식: **어떤 작업 실패에도 종료하지 않는다.** 종료하면 GPU 가 논다.
    """

    def __init__(
        self,
        spec: QueueSpec,
        *,
        dry_run: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        proc_root: str | Path = "/proc",
        log: Callable[[str], None] | None = None,
        exit_when_drained: bool = False,
    ) -> None:
        self.spec = spec
        self.dry_run = dry_run
        self.sleep = sleep
        self.proc_root = proc_root
        self.exit_when_drained = exit_when_drained
        self._log = log or (lambda message: print(message, flush=True))

        state_dir = _abs(spec.state_dir)
        self.state_dir = state_dir
        self.log_dir = state_dir / "logs"
        self.status = StatusWriter(
            state_dir / f"gpu{spec.gpu}.json", state_dir / "events.jsonl"
        )
        self.results: dict[str, dict] = {}
        self.idle_seconds_total = 0.0
        self._held_locks: list[FileLock] = []
        self._owned_pgids: set[int] = set()
        self._active: subprocess.Popen | None = None
        self._active_job: Job | None = None

    # -- 로깅 -------------------------------------------------------------

    def log(self, message: str) -> None:
        self._log(f"[{kst_now_str()} KST] {message}")

    # -- 진입 게이트 -------------------------------------------------------

    def acquire_own_lock(self) -> FileLock:
        lock = FileLock(
            _abs("runs") / f".job_queue_gpu{self.spec.gpu}.lock",
            role=f"job-queue gpu{self.spec.gpu}",
            metadata={"queue": self.spec.source},
        )
        lock.acquire()
        self._held_locks.append(lock)
        return lock

    def wait_entry_gate(self) -> None:
        gate = self.spec.entry_gate or {}
        poll = self.spec.tunable("poll_seconds", 20.0)

        for entry in gate.get("wait_for_pids") or []:
            pid = int(entry["pid"])
            patterns = list(entry.get("cmdline_contains") or [])
            note = entry.get("note", "")
            self.status.set(state="waiting_entry_gate", waiting_on={"pid": pid, "note": note})

            def _report(identity: dict, _pid: int = pid, _note: str = note) -> None:
                self.log(f"진입 대기: pid {_pid} 생존 중 ({_note})")
                self.status.set(
                    state="waiting_entry_gate",
                    waiting_on={"pid": _pid, "note": _note, "cmdline": identity["cmdline"]},
                )

            outcome = wait_for_pid_exit(
                pid,
                patterns,
                poll_seconds=poll,
                on_poll=_report,
                sleep=self.sleep,
                proc_root=self.proc_root,
            )
            self.log(f"진입 게이트: pid {pid} → {outcome}")
            self.status.event("entry_gate_pid", pid=pid, outcome=outcome)

        for lock_path in gate.get("acquire_locks") or []:
            target = _abs(lock_path)
            while True:
                lock = FileLock(target, role=f"job-queue gpu{self.spec.gpu} (inherited)")
                try:
                    lock.acquire()
                except LockHeldError as exc:
                    self.log(f"진입 대기: 기존 lock 보유 중 — {exc}")
                    self.status.set(state="waiting_entry_gate", waiting_on={"lock": str(target)})
                    self.sleep(poll)
                    continue
                self._held_locks.append(lock)
                self.log(f"진입 게이트: lock 획득 {target}")
                break

        if gate.get("require_gpu_free", True):
            self.wait_gpu_free()

    def wait_gpu_free(self) -> None:
        needed = self.spec.tunable("gpu_free_confirmations", 3)
        poll = self.spec.tunable("gpu_free_poll_seconds", 10.0)
        settle = self.spec.tunable("settle_seconds", 15.0)
        if settle > 0:
            self.sleep(settle)
        confirmations = 0
        while confirmations < needed:
            free, reason = gpu_is_free(
                self.spec.gpu,
                memory_threshold_mib=self.spec.tunable("gpu_free_memory_threshold_mib", 1024),
                proc_root=self.proc_root,
            )
            if free:
                confirmations += 1
                self.log(f"GPU{self.spec.gpu} 유휴 확인 {confirmations}/{needed} — {reason}")
            else:
                if confirmations:
                    self.log(f"GPU{self.spec.gpu} 재점유 감지 — 확인 초기화 ({reason})")
                confirmations = 0
                self.status.set(state="waiting_gpu_free", waiting_on={"gpu": reason})
            if confirmations < needed:
                self.sleep(poll)

    # -- 자식 프로세스 ------------------------------------------------------

    def terminate_group(self, pgid: int) -> None:
        """자신이 만든 프로세스 그룹만 종료할 수 있다.

        기존에 돌고 있는 학습/watcher 에는 어떤 신호도 보낼 수 없게 하는 방어선이다.
        """

        if pgid not in self._owned_pgids:
            raise ValueError(f"감독자가 소유하지 않은 pgid 에 신호를 보낼 수 없습니다: {pgid}")
        grace = self.spec.tunable("term_grace_seconds", 20.0)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            self.sleep(1.0)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def await_exit(self, process: subprocess.Popen, timeout: float) -> bool:
        """자식이 끝날 때까지 최대 ``timeout`` 초 블로킹 대기. 끝났으면 True.

        ``poll() + sleep`` 스핀 대신 OS 대기를 쓴다. 스핀은 폴링 간격이 0일 때 CPU 를
        태우고, 학습 DataLoader 가 32 vCPU 중 28개를 쓰는 원격에서는 그대로 손해다.
        """

        try:
            process.wait(timeout=max(float(timeout), 0.05))
        except subprocess.TimeoutExpired:
            return False
        return True

    def child_env(self) -> dict[str, str]:
        """자식 프로세스 환경 — ``CUDA_VISIBLE_DEVICES`` 를 반드시 이 GPU 로 고정한다.

        이걸 빠뜨리면 자식이 두 GPU 를 모두 보고 PyTorch 가 기본값 cuda:0 에 올라간다.
        GPU1 감독자가 띄운 학습이 GPU0 의 다른 학습 위에 겹쳐서 메모리를 뺏고 속도를
        떨어뜨린다(실제로 한 번 발생시켰다). ``processes_using_gpu`` 의 유휴 판정도 이
        환경변수를 근거로 하므로, 설정하지 않으면 감독자끼리 서로를 못 본다.
        """

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu)
        return env

    def spawn(self, command: Sequence[str], log_path: Path) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603 — 명령은 큐 정의에서만 온다
            list(command),
            cwd=str(REPO_ROOT),
            env=self.child_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        self._owned_pgids.add(os.getpgid(process.pid))
        handle.close()
        return process

    # -- 명령 구성 ---------------------------------------------------------

    def train_command(self, job: Job) -> list[str]:
        command = [
            self.spec.python,
            "scripts/train/train.py",
            "--config",
            job.config,
        ]
        if job.resume:
            command += ["--resume", job.resume]
        overrides = dict(job.overrides)
        if job.model_config:
            overrides.setdefault("model_config", job.model_config)
        if job.ckpt_dir:
            overrides["ckpt_dir"] = job.ckpt_dir
        for key in sorted(overrides):
            command += ["--set", f"{key}={_yaml_scalar(overrides[key])}"]
        return command

    def eval_command(self, job: Job, ckpt: str, out: str, n_items: int) -> list[str]:
        run_dir = _abs(job.ckpt_dir or ".")
        return [
            self.spec.python,
            "scripts/eval/evaluate_offline.py",
            "--ckpt",
            str(run_dir / "ckpt" / f"{ckpt}.pt"),
            "--n-items",
            str(int(n_items)),
            "--out",
            str(run_dir / out),
        ]

    # -- 작업 실행 ---------------------------------------------------------

    def restore_results(self) -> None:
        """이전 감독자가 남긴 작업 결과를 불러온다.

        결과를 메모리에만 두면 재기동할 때마다 완료된 작업을 처음부터 다시 돌린다.
        실제로 완주한 20k 대조군을 재실행해 checkpoint 를 덮어쓸 뻔했다(step 500 직전에
        차단). ``running`` 처럼 종료되지 않은 상태는 복원하지 않아 정상적으로 재시도된다.
        """

        previous = read_status(self.status.path)
        if not previous:
            return
        terminal = {"succeeded", "adopted", "already_done"}
        restored = {}
        for job_id, record in (previous.get("jobs") or {}).items():
            state = record.get("state")
            if state in terminal or str(state).startswith(("failed", "skipped")):
                restored[job_id] = record
        if restored:
            self.results.update(restored)
            self.log(f"이전 결과 복원: {sorted(restored)}")
        self.idle_seconds_total = float(previous.get("idle_seconds_total") or 0.0)
        for key in ("decisions",):
            if previous.get(key):
                self.status.data[key] = previous[key]

    def run_job(self, job: Job) -> dict:
        if job.kind == "adopt":
            return self.verify_run(job.id, job.ckpt_dir or "", job.expect)
        # 상태 파일이 유실됐더라도 디스크가 이미 완료를 증명하면 다시 돌리지 않는다.
        # 학습 재실행은 시간 낭비일 뿐 아니라 완주 checkpoint 를 덮어쓰는 파괴적 동작이다.
        if job.kind == "train" and job.ckpt_dir and job.expect:
            done = self.verify_run(job.id, job.ckpt_dir, job.expect)
            if done["state"] == "succeeded":
                self.log(f"[{job.id}] 이미 완료돼 있어 건너뛴다 ({job.ckpt_dir})")
                return self.outcome(
                    job, "already_done", f"디스크에 완료 상태로 존재: {job.ckpt_dir}",
                    checkpoint=done.get("checkpoint"),
                )
        if job.kind == "decision":
            return self.run_decision(job)
        if job.kind == "bundle":
            return self.run_bundle(job)

        precondition = self.check_preconditions(job)
        if precondition is not None:
            return precondition

        attempts = 0
        allowed = job.max_retries if job.on_failure == "retry_then_continue" else 0
        while True:
            attempts += 1
            outcome = self.run_attempt(job, attempts)
            state = outcome["state"]
            if state == "succeeded":
                return outcome
            retryable = state in {"failed_transient", "failed_killed"}
            if retryable and attempts <= allowed:
                self.log(f"[{job.id}] {state} — 동일 설정으로 재시도 {attempts}/{allowed}")
                self.preserve_failed(job, attempts)
                continue
            return outcome

    def check_preconditions(self, job: Job) -> dict | None:
        free_gb = shutil.disk_usage(str(REPO_ROOT)).free / (1024**3)
        min_free = self.spec.tunable("min_free_gb", 10.0)
        if free_gb < min_free:
            return self.outcome(job, "skipped_precondition", f"디스크 여유 {free_gb:.1f}GB < {min_free}GB")
        if job.kind == "train" and job.ckpt_dir:
            # **GPU 를 가리지 않고** 같은 ckpt_dir 을 쓰는 학습을 찾는다. 자기 GPU 만 보면
            # 다른 GPU 감독자가 같은 run 을 돌리는 것을 놓쳐 checkpoint 를 서로 덮어쓴다
            # (실제로 GPU0 이 GPU1 의 완료 run 을 덮어썼다).
            busy = [
                p
                for p in processes_with_ckpt_dir(job.ckpt_dir, proc_root=self.proc_root)
                if p["pid"] != os.getpid()
            ]
            if busy:
                return self.outcome(
                    job,
                    "skipped_precondition",
                    f"같은 ckpt_dir 을 다른 학습이 쓰는 중: pid {busy[0]['pid']}",
                )
        for item in job.copy_before_start:
            source = _abs(item["src"])
            destination = _abs(item["dst"])
            if source.exists() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                self.log(f"[{job.id}] 사전 복사: {source} → {destination}")
        return None

    def run_attempt(self, job: Job, attempt: int) -> dict:
        if job.kind == "train":
            command = self.train_command(job)
        elif job.kind == "shell":
            command = [str(part) for part in job.command]
        else:
            command = [str(part) for part in job.command]
        log_path = self.log_dir / f"{job.id}.{attempt}.log"

        if self.dry_run:
            self.log(f"[dry-run] {job.id}: {' '.join(command)}")
            return self.outcome(job, "dry_run", "실행하지 않음", command=command)

        self.log(f"[{job.id}] 시작 (attempt {attempt}): {' '.join(command)}")
        started = time.monotonic()
        process = self.spawn(command, log_path)
        self._active, self._active_job = process, job
        self.status.set(
            state="running",
            current_job={
                "id": job.id,
                "attempt": attempt,
                "pid": process.pid,
                "log": str(log_path),
                "command": command,
                "started_at_utc": utc_now_iso(),
            },
        )

        interval = self.spec.tunable("status_interval_seconds", 30.0)
        # 상태 기록은 fsync 를 동반하므로 반드시 벽시계로 스로틀한다. interval 이 0이면
        # 이 루프가 타이트 스핀이 되고, 그때 매 반복 fsync + 로그 재파싱을 하면 사실상 멈춘다.
        write_every = max(float(interval), 1.0)
        next_write = time.monotonic() + write_every
        while not self.await_exit(process, interval):
            if time.monotonic() < next_write:
                continue
            next_write = time.monotonic() + write_every
            progress = read_progress(log_path)
            self.status.set(
                current_job={
                    **(self.status.data.get("current_job") or {}),
                    "step": progress.get("step"),
                    "rate_it_s": progress.get("rate_it_s"),
                }
            )
        returncode = process.returncode
        self._active, self._active_job = None, None
        elapsed = time.monotonic() - started

        state = classify_failure(returncode, log_path)
        if state == "succeeded" and job.kind == "train":
            verified = self.verify_run(job.id, job.ckpt_dir or "", job.expect)
            if verified["state"] != "succeeded":
                return self.outcome(
                    job, "failed_validation", verified["detail"], attempt=attempt,
                    returncode=returncode, elapsed_seconds=elapsed, log=str(log_path),
                )
            if job.post_eval:
                self.run_post_eval(job)
        return self.outcome(
            job, state, f"exit {returncode}", attempt=attempt, returncode=returncode,
            elapsed_seconds=elapsed, log=str(log_path),
        )

    def run_post_eval(self, job: Job) -> None:
        """평가 실패는 학습 성과를 무효화하지 않는다 — 기록만 하고 계속 간다."""

        for index, item in enumerate(job.post_eval):
            ckpt = str(item.get("ckpt", "last"))
            out = str(item.get("out", f"eval_{ckpt}"))
            n_items = int(item.get("n_items", 32))
            command = self.eval_command(job, ckpt, out, n_items)
            log_path = self.log_dir / f"{job.id}.eval_{ckpt}.{index}.log"
            if self.dry_run:
                self.log(f"[dry-run] {job.id} 평가: {' '.join(command)}")
                continue
            self.log(f"[{job.id}] 평가 {ckpt} 시작")
            process = self.spawn(command, log_path)
            self._active = process
            while not self.await_exit(process, self.spec.tunable("status_interval_seconds", 30.0)):
                pass
            self._active = None
            self.log(f"[{job.id}] 평가 {ckpt} 종료 (exit {process.returncode})")
            if process.returncode != 0:
                self.log(f"[{job.id}] 평가 {ckpt} 실패 (exit {process.returncode}) — 계속 진행")
            self.status.event(
                "post_eval", job=job.id, ckpt=ckpt, returncode=process.returncode, log=str(log_path)
            )

    def preserve_failed(self, job: Job, attempt: int) -> None:
        """실패 산출물을 지우지 않고 옮겨 보존한다(같은 파일시스템 → 원자적 rename)."""

        if not job.ckpt_dir:
            return
        source = _abs(job.ckpt_dir)
        if not source.exists():
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = _abs("runs/failed") / f"{job.id}_{attempt}_{stamp}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(destination)
            self.log(f"[{job.id}] 실패 산출물 보존: {destination}")
        except OSError as exc:
            self.log(f"[{job.id}] 실패 산출물 보존 실패(무시): {exc}")

    def verify_run(self, job_id: str, ckpt_dir: str, expect: dict) -> dict:
        run_dir = _abs(ckpt_dir)
        last = run_dir / "ckpt" / "last.pt"
        info = checkpoint_info(last)
        if info is None:
            return {"id": job_id, "state": "failed_validation", "detail": f"last.pt 판독 불가: {last}"}
        for key, wanted in (expect or {}).items():
            actual = info.get(key)
            if actual != wanted:
                return {
                    "id": job_id,
                    "state": "failed_validation",
                    "detail": f"{key} 불일치: 기대 {wanted!r}, 실제 {actual!r}",
                    "checkpoint": info,
                }
        best = run_dir / "ckpt" / "best.pt"
        if not best.exists():
            return {"id": job_id, "state": "failed_validation", "detail": f"best.pt 없음: {best}"}
        return {"id": job_id, "state": "succeeded", "detail": "검증 통과", "checkpoint": info}

    def run_decision(self, job: Job) -> dict:
        from .structure_select import decide_structure_winner  # noqa: PLC0415

        try:
            verdict = decide_structure_winner(job.decision, log=self.log)
        except Exception as exc:  # noqa: BLE001 — 결정 실패로 큐를 멈추지 않는다
            return self.outcome(job, "failed_deterministic", f"결정 실패: {exc}")
        self.status.set(decisions={**(self.status.data.get("decisions") or {}), job.id: verdict})
        self.status.event("decision", job=job.id, verdict=verdict)
        emitted = self.emit_from_decision(job, verdict)
        state = "succeeded" if verdict.get("winner") else "failed_validation"
        detail = verdict.get("summary", "")
        if emitted:
            detail += f" → 후속 작업 {emitted} 추가"
        return self.outcome(job, state, detail, verdict=verdict, emitted=emitted)

    def emit_from_decision(self, job: Job, verdict: dict) -> list[str]:
        """결정 결과로 후속 작업을 큐에 넣는다.

        이게 없으면 결정 직후 큐가 말라 GPU 가 논다 — 감독자를 만든 이유 자체가 사라진다.
        """

        template = (job.decision or {}).get("extension_template")
        if not template:
            return []
        if verdict.get("ambiguous"):
            self.log("winner_ambiguous — 연장을 자동 시작하지 않는다(seed 반복 결과를 먼저 본다)")
            return []
        if verdict.get("winner_is_control"):
            self.log(
                "승자가 대조군이다 — tiny 100k 완주본이 이미 있으므로 연장하지 않는다"
            )
            return []
        winner = verdict.get("winner")
        if not winner:
            return []
        mapping = {
            "winner": winner,
            "winner_run_dir": verdict.get("winner_run_dir") or "",
            "winner_model_config": verdict.get("winner_model_config") or "",
        }
        try:
            spec = Job(**_substitute(template, mapping))
        except TypeError as exc:
            self.log(f"연장 작업 생성 실패(큐는 계속 진행): {exc}")
            return []
        if any(existing.id == spec.id for existing in self.spec.jobs):
            return []
        self.spec.jobs.append(spec)
        self.log(f"연장 작업 추가: {spec.id} (승자 {winner})")
        self.status.event("job_emitted", job=spec.id, source=job.id, winner=winner)
        return [spec.id]

    def run_bundle(self, job: Job) -> dict:
        """회수 대상 산출물의 SHA-256/크기/경로와 scp 명령을 기록한다."""

        import hashlib  # noqa: PLC0415

        entries: list[dict] = []
        for pattern in job.command or ["runs/*/ckpt/*.pt"]:
            for path in sorted(REPO_ROOT.glob(str(pattern))):
                if not path.is_file():
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda h=handle: h.read(1 << 20), b""):
                        digest.update(block)
                entries.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
        payload = {"generated_at_utc": utc_now_iso(), "artifacts": entries}
        atomic_write_text(
            self.state_dir / "handoff.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return self.outcome(job, "succeeded", f"회수 목록 {len(entries)}건 기록")

    def outcome(self, job: Job, state: str, detail: str, **extra: Any) -> dict:
        record = {"id": job.id, "kind": job.kind, "tier": job.tier, "state": state,
                  "detail": detail, "at_utc": utc_now_iso(), **extra}
        self.status.event("job_result", **record)
        return record

    # -- 메인 루프 ---------------------------------------------------------

    def run(self) -> int:
        # 복원이 반드시 첫 기록보다 앞서야 한다. 순서가 바뀌면 jobs={} 로 이전 결과를
        # 덮어쓴 뒤 그 빈 파일을 읽게 되어 복원이 조용히 무력화된다.
        self.restore_results()
        self.status.set(
            state="starting",
            gpu=self.spec.gpu,
            pid=os.getpid(),
            host=socket.gethostname(),
            queue_file=self.spec.source,
            started_at_utc=utc_now_iso(),
            idle_seconds_total=round(self.idle_seconds_total, 1),
            jobs=dict(self.results),
            # 리더가 STALE 을 판정하려면 감독자가 얼마나 자주 쓰는지 알아야 한다.
            # drained 상태에서는 300초마다 쓰므로 30초 기준으로 보면 항상 STALE 로 보인다.
            status_interval_seconds=self.spec.tunable("status_interval_seconds", 30.0),
            drained_poll_seconds=self.spec.tunable("drained_poll_seconds", 300.0),
        )
        self.log(f"감독자 시작: GPU{self.spec.gpu}, 큐 {self.spec.source}")

        self.wait_entry_gate()
        self.log("진입 게이트 통과 — 작업 시작")

        for entry in self.spec.adopt or []:
            result = self.verify_run(
                str(entry.get("id")), str(entry.get("run_dir", "")), entry.get("expect") or {}
            )
            result["state"] = "adopted" if result["state"] == "succeeded" else result["state"]
            self.results[result["id"]] = result
            self.log(f"[adopt] {result['id']}: {result['state']} — {result['detail']}")
            self.record_jobs()

        idle_started: float | None = None
        while True:
            job = self.next_job()
            if job is None:
                if idle_started is None:
                    idle_started = time.monotonic()
                    self.log("실행할 Tier-A 작업이 없습니다 — 큐 파일을 계속 재확인합니다")
                self.status.set(
                    state="drained",
                    recommendation="teardown",
                    idle_seconds_total=self.idle_seconds_total,
                )
                if self.exit_when_drained:
                    return 0
                self.sleep(self.spec.tunable("drained_poll_seconds", 300.0))
                self.idle_seconds_total += self.spec.tunable("drained_poll_seconds", 300.0)
                self.reload_queue()
                continue

            if idle_started is not None:
                self.idle_seconds_total += time.monotonic() - idle_started
                idle_started = None
            self.status.set(idle_seconds_total=round(self.idle_seconds_total, 1))

            result = self.run_job(job)
            self.results[job.id] = result
            self.log(f"[{job.id}] {result['state']} — {result['detail']}")
            self.record_jobs()
            self.reload_queue()

    def next_job(self) -> Job | None:
        for job in self.spec.jobs:
            if job.id in self.results:
                continue
            unmet = [
                d
                for d in job.depends_on
                if self.results.get(d, {}).get("state") not in {"succeeded", "adopted", "already_done"}
            ]
            if unmet:
                if all(d in self.results for d in job.depends_on):
                    self.results[job.id] = {
                        "id": job.id, "kind": job.kind, "state": "skipped_dependency",
                        "detail": f"선행 작업 실패: {unmet}", "at_utc": utc_now_iso(),
                    }
                    self.log(f"[{job.id}] skipped_dependency — 선행 작업 실패: {unmet}")
                continue
            return job
        return None

    def reload_queue(self) -> None:
        """큐 파일을 다시 읽어 새로 추가된 작업을 반영한다.

        감독자를 재시작하지 않고도 유휴가 예상되는 GPU 에 작업을 덧붙일 수 있다.
        이미 결과가 있는 작업 id 는 건너뛰므로 재실행되지 않는다.
        """

        try:
            fresh = load_queue(self.spec.source)
        except (QueueSpecError, OSError) as exc:
            self.log(f"큐 재로드 실패(기존 큐 유지): {exc}")
            return
        known = {job.id for job in self.spec.jobs}
        added = [job for job in fresh.jobs if job.id not in known]
        if added:
            self.spec.jobs.extend(added)
            self.log(f"큐 재로드: 신규 작업 {[j.id for j in added]}")
            self.status.event("queue_reload", added=[j.id for j in added])

    def record_jobs(self) -> None:
        self.status.set(jobs=dict(self.results))

    def close(self) -> None:
        for lock in reversed(self._held_locks):
            lock.release()
        self._held_locks.clear()


def _substitute(value: Any, mapping: dict[str, str]) -> Any:
    """중첩 구조 안의 ``{winner}`` 같은 자리표시자를 치환한다."""

    if isinstance(value, str):
        out = value
        for key, replacement in mapping.items():
            out = out.replace("{" + key + "}", str(replacement))
        return out
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    return value


def _yaml_scalar(value: Any) -> str:
    """``--set key=value`` 에 넣을 스칼라 표현. YAML 로 다시 읽히므로 그대로 쓴다."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


__all__ = [
    "FileLock",
    "Job",
    "LockHeldError",
    "QueueSpec",
    "QueueSpecError",
    "StatusWriter",
    "Supervisor",
    "atomic_write_text",
    "checkpoint_info",
    "classify_failure",
    "gpu_is_free",
    "gpu_snapshot",
    "identity_matches",
    "load_queue",
    "lock_is_held",
    "lock_owner",
    "proc_identity",
    "processes_using_gpu",
    "read_progress",
    "read_status",
    "validate_comparability",
    "wait_for_pid_exit",
]
