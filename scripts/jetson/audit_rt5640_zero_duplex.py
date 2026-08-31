#!/usr/bin/env python3
"""RT5640/APE exact-zero duplex의 read-only 현장 어댑터.

기본 동작은 장치를 열지 않는 dry-run이다. ``--execute-live``는 clean exact commit,
현재 worktree ``src`` binding, 다섯 개 물리 확인, Pulse APE profile off, 모든 APE PCM
무점유를 확인한 뒤에만 APE pcm1(input)/pcm0(output)을 동시에 연다. 출력 callback은
``deep_anc.audio_zero_duplex``가 제공하는 bitwise zero 이외의 값을 받을 수 없다.

이 smoke는 shared clock, hardware sample identity/drop/add, P/S, attenuation 또는 학습
권한을 만들지 않는다. ALSA/Pulse/mixer도 변경하지 않는다. 실행 전후 read-only snapshot이
다르면 외부 상태가 바뀐 것으로 보고 ``STATE_UNCERTAIN``으로 실패하며 자동 복구하지 않는다.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
ADAPTER_RELATIVE_PATH = "scripts/jetson/audit_rt5640_zero_duplex.py"
CONFIG_RELATIVE_PATH = "configs/hardware_jetson_rt5640_zero_smoke.yaml"
PROC_ASOUND_ROOT = Path("/proc/asound")
DEV_SND_ROOT = Path("/dev/snd")
SYS_CLASS_SOUND_ROOT = Path("/sys/class/sound")
SYS_ROOT = Path("/sys")
RT5640_I2C_DEVICE = Path("/sys/bus/i2c/devices/8-001c")
PULSE_APE_CARD_NAME = "alsa_card.platform-sound"
GLOBAL_LOCK_BASENAME = "deep_anc-machine-live-audio.lock"

LIVE_RECEIPT_SCHEMA = "rt5640_zero_duplex_live_receipt_v1"
FAILURE_RECEIPT_SCHEMA = "rt5640_zero_duplex_live_failure_v1"
CLAIM_SCHEMA = "rt5640_zero_duplex_generation_claim_v1"
SNAPSHOT_SCHEMA = "rt5640_zero_duplex_read_only_snapshot_v1"
EXPECTED_AUTHORITY_CEILING = "ZERO_DUPLEX_TRANSPORT_SMOKE_PASS"

_REQUIRED_LIVE_FLAGS = (
    "confirm_j511_disconnected",
    "confirm_amplifier_power_off",
    "confirm_amplifier_input_disconnected",
    "confirm_ab13x_amplifier_disconnected",
    "confirm_user_present",
)

_TOOL_VERSION_ARGS = {
    "git": ("--version",),
    "fuser": ("--version",),
    "pactl": ("--version",),
    "alsactl": ("--version",),
    "amixer": ("--version",),
}

_VOLATILE_CONTROL_NAMES = {
    1132: "Lane1 Ratio Int",
    1133: "Lane1 Ratio Frac",
    1134: "Lane2 Ratio Int",
    1135: "Lane2 Ratio Frac",
    1136: "Lane3 Ratio Int",
    1137: "Lane3 Ratio Frac",
    1138: "Lane4 Ratio Int",
    1139: "Lane4 Ratio Frac",
    1140: "Lane5 Ratio Int",
    1141: "Lane5 Ratio Frac",
    1142: "Lane6 Ratio Int",
    1143: "Lane6 Ratio Frac",
}
_VOLATILE_SENTINEL = "__RT5640_READ_VOLATILE_VALUE__"


class StateUncertainError(RuntimeError):
    """오디오 종료 뒤 ALSA state/mixer가 baseline과 달라진 경우."""


class DeferredSignalError(RuntimeError):
    """read-only transaction의 close/post-verify 뒤 전달하는 종료 신호."""

    def __init__(self, signum: int):
        super().__init__(f"signal {signum} received during live transaction")
        self.signum = int(signum)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> dict[str, Any]:
    """regular file을 O_EXCL/no-follow로 쓰고 parent까지 fsync한다."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive evidence write가 전진하지 않았습니다")
            view = view[written:]
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != len(payload):
            raise RuntimeError("exclusive evidence가 regular/exact-size가 아닙니다")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    observed = path.read_bytes()
    if observed != payload:
        raise RuntimeError(f"published evidence bytes가 바뀌었습니다: {path}")
    return {
        "path": str(path),
        "size": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _write_exclusive(path, _canonical_json_bytes(dict(value)))


def _publish_final_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """완성·fsync된 sibling inode를 hardlink해 final JSON을 원자 no-replace 공개한다."""

    payload = _canonical_json_bytes(dict(value))
    temporary = path.parent / f".{path.name}.complete.{os.getpid()}.{time.time_ns()}"
    linked = False
    try:
        _write_exclusive(temporary, payload)
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        _fsync_directory(path.parent)
        if path.read_bytes() != payload:
            raise RuntimeError("final JSON publish 뒤 bytes가 바뀌었습니다")
        return {"path": str(path), "size": len(payload), "sha256": _sha256_bytes(payload)}
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass
        except OSError:
            if not linked:
                raise


def _publish_terminal_receipt(
    generation: Path, *, success: bool, value: Mapping[str, Any]
) -> dict[str, Any]:
    """success/failure terminal receipt가 한 generation에 공존하지 못하게 한다."""

    target_name = "receipt.json" if success else "failure.json"
    opposite_name = "failure.json" if success else "receipt.json"
    if (generation / opposite_name).exists():
        raise RuntimeError(
            f"terminal receipt 공존 금지: {opposite_name}가 이미 존재합니다"
        )
    return _publish_final_json_exclusive(generation / target_name, value)


def _publish_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """같은 generation 안에서 temp+hardlink로 NPZ를 no-replace 공개한다."""

    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("raw NPZ가 regular no-symlink file이 아닙니다")
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}


def _run_command(
    command: Sequence[str],
    *,
    timeout: float = 10.0,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    if not command:
        raise ValueError("빈 external command는 허용하지 않습니다")
    executable = shutil.which(str(command[0]))
    if not executable:
        raise RuntimeError(f"external executable을 찾지 못했습니다: {command[0]}")
    resolved_executable = Path(executable).resolve(strict=True)
    actual_command = [str(resolved_executable), *[str(item) for item in command[1:]]]
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    result = subprocess.run(
        actual_command,
        capture_output=True,
        text=True,
        timeout=float(timeout),
        check=False,
        env=environment,
    )
    if result.returncode not in accepted_returncodes:
        raise RuntimeError(
            f"command 실패 rc={result.returncode}: {' '.join(actual_command)}\n"
            f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
        )
    return result


def _tool_fingerprints() -> dict[str, Any]:
    """현장 판정에 쓰는 외부 실행 파일의 realpath/content/version을 결속한다."""

    result: dict[str, Any] = {}
    for name, version_args in _TOOL_VERSION_ARGS.items():
        located = shutil.which(name)
        if not located:
            raise RuntimeError(f"필수 executable을 찾지 못했습니다: {name}")
        path = Path(located).resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"필수 executable이 regular/executable이 아닙니다: {path}")
        version = _run_command([str(path), *version_args], timeout=5.0)
        version_text = (version.stdout + version.stderr).strip()
        if not version_text:
            raise RuntimeError(f"{name} version 출력이 비었습니다")
        result[name] = {
            "resolved_path": str(path),
            "file_sha256": _sha256_file(path),
            "version_output": version_text,
            "locale": "C",
        }
    return result


def _repo_relative_regular(path: Path, *, expected_relative: str) -> Path:
    expected = (REPO_ROOT / expected_relative).resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"sealed repository file이 symlink입니다: {path}")
    observed = path.resolve(strict=True)
    if observed != expected or not observed.is_file():
        raise ValueError(f"sealed repository file이 아닙니다: {path}")
    return observed


def _git_identity(expected_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit or ""):
        raise ValueError("--expected-commit은 lowercase 40-hex여야 합니다")
    head = _run_command(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], timeout=5.0
    ).stdout.strip()
    branch = _run_command(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"], timeout=5.0
    ).stdout.strip()
    dirty = _run_command(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        timeout=10.0,
    ).stdout
    if head != expected_commit:
        raise RuntimeError(f"expected commit 불일치: expected={expected_commit}, actual={head}")
    if dirty:
        raise RuntimeError("live 실행은 clean checkout만 허용합니다")
    if not branch:
        raise RuntimeError("detached HEAD에서는 live generation을 발행하지 않습니다")
    return {"commit": head, "branch": branch, "dirty": False}


def _resolve_pythonpath_entry(entry: str) -> Path:
    candidate = Path(entry).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=True)


def _bind_script_to_current_worktree_src() -> Path:
    """직접 CLI 실행도 같은 checkout ``src``를 최우선 import 경로로 묶는다.

    이 함수는 package/backend를 import하지 않고 ``sys.path``만 정렬한다. 따라서
    dry-run은 ``PYTHONPATH=src``라는 셸 환경 설정에 의존하지 않으면서도, 이어지는
    static 계약 import가 설치본이나 다른 checkout을 보지 않게 한다. 이미 다른
    ``deep_anc``가 preload된 경우에는 경로를 덮어써 숨기지 않고 fail-closed한다.
    """

    expected_src = (REPO_ROOT / "src").resolve(strict=True)
    package = sys.modules.get("deep_anc")
    if package is not None:
        package_file_raw = getattr(package, "__file__", None)
        if not isinstance(package_file_raw, str) or not package_file_raw:
            raise RuntimeError("preload된 deep_anc package file을 확인할 수 없습니다")
        package_file = Path(package_file_raw).resolve(strict=True)
        try:
            package_file.relative_to(expected_src)
        except ValueError as exc:
            raise RuntimeError(
                f"preload된 deep_anc import가 current worktree 밖입니다: {package_file}"
            ) from exc

    # ``python scripts/jetson/...``는 scripts/jetson만 sys.path[0]에 놓는다.
    # 직접 CLI도 source checkout이 첫 항목이어야 static import boundary가 유지된다.
    source_text = str(expected_src)
    if not sys.path or _resolve_pythonpath_entry(sys.path[0]) != expected_src:
        sys.path.insert(0, source_text)
    if _resolve_pythonpath_entry(sys.path[0]) != expected_src:
        raise RuntimeError("current worktree src를 sys.path 첫 항목으로 결속하지 못했습니다")
    return expected_src


def _assert_current_worktree_binding() -> dict[str, Any]:
    expected_src = _bind_script_to_current_worktree_src()
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    package = importlib.import_module("deep_anc")
    package_file = Path(str(package.__file__)).resolve(strict=True)
    try:
        package_file.relative_to(expected_src)
    except ValueError as exc:
        raise RuntimeError(f"deep_anc import가 current worktree 밖입니다: {package_file}") from exc
    prefix = Path(sys.prefix).resolve(strict=True)
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    if prefix == base_prefix:
        raise RuntimeError("live audit은 활성 venv Python에서만 허용합니다")
    executable_lexical = Path(sys.executable).absolute()
    if not executable_lexical.is_file():
        raise RuntimeError("sys.executable lexical path가 file이 아닙니다")
    if executable_lexical.parent.resolve(strict=True) != (prefix / "bin").resolve(strict=True):
        raise RuntimeError("sys.executable이 현재 sys.prefix/bin 아래가 아닙니다")
    executable_target = executable_lexical.resolve(strict=True)

    def package_binding(module: ModuleType, label: str) -> dict[str, Any]:
        module_path = Path(str(module.__file__)).resolve(strict=True)
        version = getattr(module, "__version__", None)
        if not module_path.is_file() or not isinstance(version, str) or not version:
            raise RuntimeError(f"{label} package path/version을 결속할 수 없습니다")
        return {
            "module_path": str(module_path),
            "module_file_sha256": _sha256_file(module_path),
            "version": version,
        }

    yaml = importlib.import_module("yaml")
    return {
        "expected_src": str(expected_src),
        "pythonpath": raw_pythonpath,
        "python_executable_lexical": str(executable_lexical),
        "python_executable_is_symlink": executable_lexical.is_symlink(),
        "python_executable_target": str(executable_target),
        "python_executable_target_sha256": _sha256_file(executable_target),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "python_version": sys.version,
        "deep_anc_file": str(package_file),
        "packages": {
            "numpy": package_binding(np, "numpy"),
            "yaml": package_binding(yaml, "yaml"),
        },
    }


def _import_zero_api() -> tuple[ModuleType, ModuleType, dict[str, Any]]:
    binding = _assert_current_worktree_binding()
    expected_src = Path(binding["expected_src"])
    audio = importlib.import_module("deep_anc.audio_zero_duplex")
    contract = importlib.import_module("deep_anc.realtime.rt5640_zero_smoke")
    audio_io = importlib.import_module("deep_anc.audio_io")
    measurement = importlib.import_module("deep_anc.dsp.measurement_level")
    module_bindings: dict[str, dict[str, str]] = {}
    for label, module in (
        ("audio", audio),
        ("contract", contract),
        ("audio_io", audio_io),
        ("measurement_level", measurement),
    ):
        module_file = Path(str(module.__file__)).resolve(strict=True)
        try:
            module_file.relative_to(expected_src)
        except ValueError as exc:
            raise RuntimeError(f"{label} module이 current worktree 밖입니다: {module_file}") from exc
        module_bindings[label] = {
            "path": str(module_file),
            "file_sha256": _sha256_file(module_file),
        }
    required_audio = ("capture_zero_duplex", "ZeroDuplexCaptureFailure")
    required_contract = (
        "build_zero_duplex_plan",
        "build_zero_duplex_receipt",
        "capture_telemetry_to_contract",
    )
    if any(not hasattr(audio, name) for name in required_audio):
        raise RuntimeError("audio_zero_duplex 공개 API가 불완전합니다")
    if any(not hasattr(contract, name) for name in required_contract):
        raise RuntimeError("rt5640_zero_smoke 공개 API가 불완전합니다")
    binding["modules"] = module_bindings
    return audio, contract, binding


def _load_and_validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = _repo_relative_regular(path, expected_relative=CONFIG_RELATIVE_PATH)
    raw = sealed.read_bytes()
    yaml = importlib.import_module("yaml")
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "rt5640_zero_duplex_hardware_v1":
        raise ValueError("RT5640 zero smoke config schema가 다릅니다")
    audio = value.get("audio")
    smoke = value.get("zero_duplex_smoke")
    if not isinstance(audio, dict) or not isinstance(smoke, dict):
        raise ValueError("config audio/zero_duplex_smoke mapping이 필요합니다")
    expected_audio = {
        "sample_rate": 48_000,
        "block_size": 256,
        "latency": "low",
    }
    for key, expected in expected_audio.items():
        if audio.get(key) != expected or (type(expected) is int and type(audio.get(key)) is not int):
            raise ValueError(f"config audio.{key}가 exact {expected!r}가 아닙니다")
    for direction, pcm in (("input", 1), ("output", 0)):
        endpoint = audio.get(direction)
        if not isinstance(endpoint, dict):
            raise ValueError(f"config audio.{direction}가 없습니다")
        exact = {"card": "APE", "pcm": pcm, "channels": 2, "dtype": "int32"}
        for key, expected in exact.items():
            if endpoint.get(key) != expected or (
                type(expected) is int and type(endpoint.get(key)) is not int
            ):
                raise ValueError(f"config audio.{direction}.{key} 계약이 다릅니다")
    if smoke.get("duration_seconds") != 60 or type(smoke.get("duration_seconds")) is not int:
        raise ValueError("zero smoke duration은 exact 60초여야 합니다")
    if smoke.get("expected_frames") != 2_880_000:
        raise ValueError("zero smoke frame count는 exact 2,880,000이어야 합니다")
    if smoke.get("expected_callbacks") != 11_250:
        raise ValueError("zero smoke callback count는 exact 11,250여야 합니다")
    if smoke.get("result_directory") != "results/rt5640_zero_duplex/v1":
        raise ValueError("zero smoke generation 경로가 sealed v1과 다릅니다")
    barriers = smoke.get("physical_barriers")
    required_barriers = {
        "require_j511_disconnected",
        "require_amplifier_power_off",
        "require_amplifier_input_disconnected",
        "require_ab13x_amplifier_disconnected",
    }
    if not isinstance(barriers, dict) or set(barriers) != required_barriers:
        raise ValueError("physical barrier key 집합이 다릅니다")
    if any(barriers[name] is not True for name in required_barriers):
        raise ValueError("모든 physical barrier는 exact true로 요구되어야 합니다")
    authority = smoke.get("maximum_authority")
    if not isinstance(authority, dict) or authority.get("status") != EXPECTED_AUTHORITY_CEILING:
        raise ValueError("maximum authority ceiling이 다릅니다")
    if any(value is not False for key, value in authority.items() if key != "status"):
        raise ValueError("zero smoke가 transport 이상의 권위를 요구합니다")
    return value, {
        "relative_path": CONFIG_RELATIVE_PATH,
        "file_sha256": _sha256_bytes(raw),
        "size": len(raw),
    }


def _require_live_flags(args: argparse.Namespace) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in _REQUIRED_LIVE_FLAGS if getattr(args, name) is not True]
    if missing:
        raise RuntimeError("live physical 확인 플래그가 부족합니다: " + ", ".join(missing))
    if not args.expected_commit:
        raise RuntimeError("--execute-live에는 --expected-commit이 필요합니다")


def _physical_confirmations(args: argparse.Namespace) -> dict[str, bool]:
    confirmations = {name: getattr(args, name) for name in _REQUIRED_LIVE_FLAGS}
    if any(value is not True for value in confirmations.values()):
        raise RuntimeError("physical confirmations는 모두 exact true여야 합니다")
    return confirmations


def _claim_generation(path: Path, claim: Mapping[str, Any]) -> dict[str, Any]:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("generation parent가 repository 밖입니다") from exc
    path.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(parent)
    return _write_json_exclusive(path / "generation_claim.json", claim)


def _parse_alsa_cards(text: str, card_id: str) -> int:
    matches = []
    pattern = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]", re.MULTILINE)
    for number, observed in pattern.findall(text):
        if observed.strip() == card_id:
            matches.append(int(number))
    if len(matches) != 1:
        raise RuntimeError(f"ALSA card {card_id!r}가 exact 하나가 아닙니다: {matches}")
    return matches[0]


def _read_text_strict(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"필수 proc evidence file이 아닙니다: {path}")
    return path.read_text(encoding="utf-8", errors="strict")


def _all_pcm_closed() -> dict[str, Any]:
    statuses = sorted(PROC_ASOUND_ROOT.glob("card*/pcm*/sub*/status"))
    if not statuses:
        raise RuntimeError("system PCM status node를 찾지 못했습니다")
    observed: dict[str, str] = {}
    busy: list[str] = []
    for path in statuses:
        text = _read_text_strict(path)
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        relative = str(path.relative_to(PROC_ASOUND_ROOT))
        observed[relative] = first
        if first != "closed":
            busy.append(f"{relative}:{first or 'empty'}")
    if busy:
        raise RuntimeError("system PCM이 점유 중입니다: " + ", ".join(busy))
    return {"status_count": len(observed), "statuses": observed}


def _all_pcm_device_nodes(card_index: int) -> list[Path]:
    nodes = sorted(DEV_SND_ROOT.glob("pcm*"))
    if not nodes:
        raise RuntimeError("system /dev/snd PCM node를 찾지 못했습니다")
    exact_input = DEV_SND_ROOT / f"pcmC{card_index}D1c"
    exact_output = DEV_SND_ROOT / f"pcmC{card_index}D0p"
    for path in (exact_input, exact_output):
        if not path.exists() or path.is_symlink():
            raise RuntimeError(f"필수 APE endpoint node가 없습니다: {path}")
    if DEV_SND_ROOT == Path("/dev/snd"):
        for path in nodes:
            if not stat.S_ISCHR(path.stat(follow_symlinks=False).st_mode):
                raise RuntimeError(f"ALSA endpoint가 character device가 아닙니다: {path}")
    return nodes


def _fuser_no_owners(nodes: Sequence[Path]) -> dict[str, Any]:
    result = _run_command(
        ["fuser", "-v", *[str(path) for path in nodes]],
        timeout=10.0,
        accepted_returncodes=frozenset({0, 1}),
    )
    if result.returncode == 0:
        raise RuntimeError(
            "fuser가 system PCM owner를 찾았습니다:\n"
            + result.stdout[-4000:]
            + result.stderr[-4000:]
        )
    if result.stdout.strip() or result.stderr.strip():
        raise RuntimeError("fuser no-owner 결과에 해석되지 않은 출력이 있습니다")
    return {"node_count": len(nodes), "returncode": 1, "owners": []}


def _pulse_ape_off(card_index: int) -> dict[str, Any]:
    cards = _run_command(["pactl", "list", "cards"], timeout=10.0).stdout
    blocks = re.split(r"(?m)^Card #\d+\s*$", cards)
    matches = [block for block in blocks if f"Name: {PULSE_APE_CARD_NAME}" in block]
    if len(matches) != 1:
        raise RuntimeError("PulseAudio APE card block이 exact 하나가 아닙니다")
    block = matches[0]
    if f'alsa.card = "{card_index}"' not in block:
        raise RuntimeError("PulseAudio APE card index가 /proc/asound와 다릅니다")
    profile = re.search(r"(?m)^\s*Active Profile:\s*(\S+)\s*$", block)
    if profile is None or profile.group(1) != "off":
        raise RuntimeError("PulseAudio APE profile이 exact off가 아닙니다; 자동 변경하지 않습니다")
    for kind in ("sinks", "sources"):
        listing = _run_command(["pactl", "list", kind], timeout=10.0).stdout
        for section in re.split(rf"(?m)^{kind[:-1].capitalize()} #\d+\s*$", listing):
            if f'alsa.card = "{card_index}"' not in section:
                continue
            state_match = re.search(r"(?m)^\s*State:\s*(\S+)\s*$", section)
            state = "UNKNOWN" if state_match is None else state_match.group(1).upper()
            if state == "RUNNING":
                raise RuntimeError(f"PulseAudio APE {kind[:-1]}가 RUNNING입니다")
    return {
        "card_name": PULSE_APE_CARD_NAME,
        "alsa_card_index": card_index,
        "active_profile": "off",
        "mutation_performed": False,
    }


def _read_route_contract() -> dict[str, Any]:
    """APE mux를 cget만으로 읽고 현 simultaneous 경로를 exact 검증한다."""

    expected = {
        "I2S1 Mux": "ADMAIF1",
        "ADMAIF1 Mux": "I2S1",
        "ADMAIF2 Mux": "I2S2",
        "I2S2 Mux": "ADMAIF2",
    }
    controls: dict[str, Any] = {}
    for name, expected_item in expected.items():
        text = _run_command(
            ["amixer", "-c", "APE", "cget", f"name={name}"], timeout=10.0
        ).stdout
        items = {
            int(number): item
            for number, item in re.findall(r"(?m)^\s*; Item #(\d+) '([^']+)'\s*$", text)
        }
        value_match = re.search(r"(?m)^\s*: values=(\d+)\s*$", text)
        if value_match is None:
            raise RuntimeError(f"{name} current enum value를 읽지 못했습니다")
        value = int(value_match.group(1))
        observed = items.get(value)
        if observed != expected_item:
            raise RuntimeError(
                f"APE route {name}={observed!r}, expected={expected_item!r}; 자동 변경하지 않습니다"
            )
        controls[name] = {
            "enum_index": value,
            "observed": observed,
            "expected": expected_item,
            "raw_sha256": _sha256_bytes(text.encode("utf-8")),
            "read_command": "amixer cget only",
        }
    return {"passed": True, "controls": controls, "mutation_performed": False}


def assert_live_pcm_clock_preconditions(*, expected_card_index: int | None = None) -> dict[str, Any]:
    """sounddevice import/stream open 직전의 fail-closed read-only gate."""

    cards_text = _read_text_strict(PROC_ASOUND_ROOT / "cards")
    card_index = _parse_alsa_cards(cards_text, "APE")
    if expected_card_index is not None and card_index != expected_card_index:
        raise RuntimeError("pre-open APE card index가 static preflight와 달라졌습니다")
    closed = _all_pcm_closed()
    nodes = _all_pcm_device_nodes(card_index)
    fuser = _fuser_no_owners(nodes)
    pulse = _pulse_ape_off(card_index)
    route = _read_route_contract()
    return {
        "alsa_card_index": card_index,
        "all_system_pcm_closed": closed,
        "fuser": fuser,
        "pulse": pulse,
        "route": route,
    }


def _collect_alsa_physical_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    """공용 ALSA fingerprint와 RT5640 codec binding을 read-only로 결속한다."""

    measurement = importlib.import_module("deep_anc.dsp.measurement_level")
    official = measurement.collect_alsa_physical_fingerprint(
        dict(config),
        proc_asound_root=PROC_ASOUND_ROOT,
        sys_class_sound_root=SYS_CLASS_SOUND_ROOT,
        sys_root=SYS_ROOT,
    )
    name = _read_text_strict(RT5640_I2C_DEVICE / "name").strip()
    modalias = _read_text_strict(RT5640_I2C_DEVICE / "modalias").strip()
    uevent = _read_text_strict(RT5640_I2C_DEVICE / "uevent")
    driver = (RT5640_I2C_DEVICE / "driver").resolve(strict=True)
    if name != "rt5640" or "realtek,rt5640" not in modalias:
        raise RuntimeError("RT5640 I2C codec identity가 다릅니다")
    if "DRIVER=rt5640" not in uevent or "OF_COMPATIBLE_0=realtek,rt5640" not in uevent:
        raise RuntimeError("RT5640 uevent driver/compatible binding이 다릅니다")
    if driver.name != "rt5640":
        raise RuntimeError("RT5640 driver symlink가 rt5640에 bound되지 않았습니다")
    codec = {
        "i2c_device": str(RT5640_I2C_DEVICE),
        "name": name,
        "modalias": modalias,
        "uevent_sha256": _sha256_bytes(uevent.encode("utf-8")),
        "driver_realpath": str(driver),
    }
    core = {
        "alsa": official,
        "rt5640": codec,
        "authority": "physical_identity_only_not_route_or_shared_clock",
        "route_authority": False,
        "shared_clock_authority": False,
    }
    return {**core, "payload_sha256": _sha256_bytes(_canonical_json_bytes(core))}


@contextmanager
def _machine_global_audio_lock() -> Iterator[dict[str, Any]]:
    runtime_raw = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime_raw:
        raise RuntimeError("XDG_RUNTIME_DIR가 없어 machine-global audio lock을 만들 수 없습니다")
    runtime = Path(runtime_raw).resolve(strict=True)
    if not runtime.is_dir() or runtime.stat().st_uid != os.getuid():
        raise RuntimeError("XDG_RUNTIME_DIR ownership가 현재 uid와 다릅니다")
    path = runtime / GLOBAL_LOCK_BASENAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise RuntimeError("machine-global audio lock이 이미 점유 중입니다")
    try:
        status = os.fstat(descriptor)
        yield {
            "path": str(path),
            "uid": os.getuid(),
            "pid": os.getpid(),
            "device": int(status.st_dev),
            "inode": int(status.st_ino),
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _repository_live_audio_lock() -> Iterator[dict[str, Any]]:
    """기존 Deep_ANC live entry와 같은 cooperative lock domain도 함께 잡는다."""

    measurement = importlib.import_module("deep_anc.dsp.measurement_level")
    with measurement.repository_audio_lock(
        REPO_ROOT, purpose="rt5640_zero_duplex_v1"
    ) as identity:
        yield dict(identity)


class _DeferredSignalScope:
    def __init__(self) -> None:
        self.pending: list[int] = []
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_DeferredSignalScope":
        signums = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            signums.append(signal.SIGHUP)
        for signum in signums:
            self.previous[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def _handler(self, signum: int, _frame: object) -> None:
        if not self.pending:
            self.pending.append(int(signum))

    def raise_if_pending(self) -> None:
        if self.pending:
            raise DeferredSignalError(self.pending[0])

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def _single_match(pattern: str, text: str, *, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"{label} match가 exact 하나가 아닙니다: {len(matches)}")
    return matches[0]


def _normalize_alsactl_volatile_values(
    payload: bytes,
) -> tuple[bytes, dict[int, dict[str, Any]]]:
    """exact 12개 read-volatile control의 value token만 sentinel로 바꾼다."""

    text = payload.decode("utf-8", errors="strict")
    starts = list(re.finditer(r"(?m)^\tcontrol\.(\d+) \{\n", text))
    if not starts:
        raise RuntimeError("alsactl state control block이 없습니다")
    blocks: dict[int, tuple[int, int, str]] = {}
    volatile_ids: set[int] = set()
    replacements: list[tuple[int, int]] = []
    controls: dict[int, dict[str, Any]] = {}
    for index, match in enumerate(starts):
        numid = int(match.group(1))
        if numid in blocks:
            raise RuntimeError(f"alsactl duplicate control numid={numid}")
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start() : stop]
        blocks[numid] = (match.start(), stop, block)
        access_matches = re.findall(
            r"(?m)^\t\t\taccess '([^']+)'\s*$", block
        )
        if len(access_matches) != 1:
            raise RuntimeError(f"alsactl control {numid} access metadata가 exact 하나가 아닙니다")
        access = access_matches[0]
        if "volatile" in access:
            if access != "read volatile":
                raise RuntimeError(f"alsactl control {numid} volatile access가 exact하지 않습니다")
            volatile_ids.add(numid)

    expected_ids = set(_VOLATILE_CONTROL_NAMES)
    if volatile_ids != expected_ids:
        raise RuntimeError(
            "alsactl read-volatile allowlist가 다릅니다: "
            f"missing={sorted(expected_ids - volatile_ids)}, extra={sorted(volatile_ids - expected_ids)}"
        )

    for numid, expected_name in _VOLATILE_CONTROL_NAMES.items():
        start, _stop, block = blocks[numid]
        iface = _single_match(
            r"^\t\tiface (\S+)\s*$", block, label=f"alsactl {numid} iface"
        ).group(1)
        name = _single_match(
            r"^\t\tname '([^']+)'\s*$", block, label=f"alsactl {numid} name"
        ).group(1)
        value_match = _single_match(
            r"^\t\tvalue ([^\n]+)$", block, label=f"alsactl {numid} value"
        )
        access = _single_match(
            r"^\t\t\taccess '([^']+)'\s*$",
            block,
            label=f"alsactl {numid} access",
        ).group(1)
        control_type = _single_match(
            r"^\t\t\ttype (\S+)\s*$", block, label=f"alsactl {numid} type"
        ).group(1)
        count = int(
            _single_match(
                r"^\t\t\tcount (\d+)\s*$", block, label=f"alsactl {numid} count"
            ).group(1)
        )
        if (
            iface != "MIXER"
            or name != expected_name
            or access != "read volatile"
            or control_type != "INTEGER"
            or count != 1
        ):
            raise RuntimeError(f"alsactl volatile control {numid} identity/metadata가 다릅니다")
        raw_value = value_match.group(1)
        if not re.fullmatch(r"\d+", raw_value):
            raise RuntimeError(f"alsactl volatile control {numid} value가 decimal scalar가 아닙니다")
        value_start = start + value_match.start(1)
        value_stop = start + value_match.end(1)
        replacements.append((value_start, value_stop))
        controls[numid] = {
            "numid": numid,
            "name": name,
            "iface": iface,
            "type": control_type,
            "count": count,
            "access": access,
            "raw_value": int(raw_value),
        }

    canonical = text
    for start, stop in sorted(replacements, reverse=True):
        canonical = canonical[:start] + _VOLATILE_SENTINEL + canonical[stop:]
    return canonical.encode("utf-8"), controls


def _normalize_amixer_volatile_values(
    payload: bytes,
) -> tuple[bytes, dict[int, dict[str, Any]]]:
    """amixer의 동일 12개 r--v---- control value token만 sentinel로 바꾼다."""

    text = payload.decode("utf-8", errors="strict")
    starts = list(
        re.finditer(
            r"(?m)^numid=(\d+),iface=([^,]+),name='([^']+)'\s*$", text
        )
    )
    if not starts:
        raise RuntimeError("amixer control block이 없습니다")
    blocks: dict[int, tuple[int, int, str, str, str]] = {}
    volatile_ids: set[int] = set()
    replacements: list[tuple[int, int]] = []
    controls: dict[int, dict[str, Any]] = {}
    for index, match in enumerate(starts):
        numid = int(match.group(1))
        if numid in blocks:
            raise RuntimeError(f"amixer duplicate control numid={numid}")
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start() : stop]
        iface = match.group(2)
        name = match.group(3)
        blocks[numid] = (match.start(), stop, block, iface, name)
        metadata_matches = list(
            re.finditer(
                r"(?m)^\s*; type=([^,]+),access=([^,]+),values=(\d+)(?:,.*)?$",
                block,
            )
        )
        if len(metadata_matches) != 1:
            raise RuntimeError(f"amixer control {numid} metadata가 exact 하나가 아닙니다")
        access = metadata_matches[0].group(2)
        if "v" in access:
            if access != "r--v----":
                raise RuntimeError(f"amixer control {numid} volatile access가 exact하지 않습니다")
            volatile_ids.add(numid)

    expected_ids = set(_VOLATILE_CONTROL_NAMES)
    if volatile_ids != expected_ids:
        raise RuntimeError(
            "amixer volatile allowlist가 다릅니다: "
            f"missing={sorted(expected_ids - volatile_ids)}, extra={sorted(volatile_ids - expected_ids)}"
        )

    for numid, expected_name in _VOLATILE_CONTROL_NAMES.items():
        start, _stop, block, iface, name = blocks[numid]
        metadata = _single_match(
            r"^\s*; type=([^,]+),access=([^,]+),values=(\d+)(?:,.*)?$",
            block,
            label=f"amixer {numid} metadata",
        )
        control_type, access, values = metadata.groups()
        value_match = _single_match(
            r"^\s*: values=([^\n]+)$", block, label=f"amixer {numid} value"
        )
        if (
            iface != "MIXER"
            or name != expected_name
            or access != "r--v----"
            or control_type != "INTEGER"
            or int(values) != 1
        ):
            raise RuntimeError(f"amixer volatile control {numid} identity/metadata가 다릅니다")
        raw_value = value_match.group(1)
        if not re.fullmatch(r"\d+", raw_value):
            raise RuntimeError(f"amixer volatile control {numid} value가 decimal scalar가 아닙니다")
        value_start = start + value_match.start(1)
        value_stop = start + value_match.end(1)
        replacements.append((value_start, value_stop))
        controls[numid] = {
            "numid": numid,
            "name": name,
            "iface": iface,
            "type": control_type,
            "count": int(values),
            "access": access,
            "raw_value": int(raw_value),
        }

    canonical = text
    for start, stop in sorted(replacements, reverse=True):
        canonical = canonical[:start] + _VOLATILE_SENTINEL + canonical[stop:]
    return canonical.encode("utf-8"), controls


def _crosscheck_volatile_identities(
    alsactl: Mapping[int, Mapping[str, Any]],
    amixer: Mapping[int, Mapping[str, Any]],
) -> None:
    expected = set(_VOLATILE_CONTROL_NAMES)
    if set(alsactl) != expected or set(amixer) != expected:
        raise RuntimeError("alsactl/amixer volatile identity set이 allowlist와 다릅니다")
    for numid in sorted(expected):
        left = alsactl[numid]
        right = amixer[numid]
        for key in ("numid", "name", "iface", "type", "count"):
            if left[key] != right[key]:
                raise RuntimeError(f"alsactl/amixer volatile {numid} {key}가 다릅니다")


def _snapshot_read_only(generation: Path, *, label: str) -> dict[str, Any]:
    """ALSA state와 모든 mixer control을 읽어 generation 안에 durable 저장한다."""

    state_path = generation / f"ape_{label}.state"
    mixer_path = generation / f"amixer_{label}.txt"
    state_canonical_path = generation / f"ape_{label}.volatile_normalized.state"
    mixer_canonical_path = generation / f"amixer_{label}.volatile_normalized.txt"
    lock_path = generation / "alsactl_read.lock"
    if state_path.exists() or mixer_path.exists():
        raise FileExistsError(f"snapshot label 재사용 금지: {label}")
    _run_command(
        [
            "alsactl",
            "-l",
            "-O",
            str(lock_path),
            "-f",
            str(state_path),
            "store",
            "APE",
        ],
        timeout=15.0,
    )
    if not state_path.is_file() or state_path.is_symlink() or state_path.stat().st_size == 0:
        raise RuntimeError("alsactl read-only snapshot이 발행되지 않았습니다")
    with state_path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(generation)
    mixer = _run_command(["amixer", "-c", "APE", "contents"], timeout=15.0).stdout.encode(
        "utf-8", errors="strict"
    )
    if not mixer:
        raise RuntimeError("amixer contents가 비었습니다")
    _write_exclusive(mixer_path, mixer)
    state_bytes = state_path.read_bytes()
    state_canonical, state_controls = _normalize_alsactl_volatile_values(state_bytes)
    mixer_canonical, mixer_controls = _normalize_amixer_volatile_values(mixer)
    _crosscheck_volatile_identities(state_controls, mixer_controls)
    _write_exclusive(state_canonical_path, state_canonical)
    _write_exclusive(mixer_canonical_path, mixer_canonical)
    volatile_controls = {
        str(numid): {
            "numid": numid,
            "name": _VOLATILE_CONTROL_NAMES[numid],
            "iface": "MIXER",
            "type": "INTEGER",
            "count": 1,
            "alsactl_access": "read volatile",
            "amixer_access": "r--v----",
            "alsactl_raw_value": state_controls[numid]["raw_value"],
            "amixer_raw_value": mixer_controls[numid]["raw_value"],
        }
        for numid in sorted(_VOLATILE_CONTROL_NAMES)
    }
    return {
        "schema": SNAPSHOT_SCHEMA,
        "label": label,
        "alsactl": {
            "path": state_path.name,
            "sha256": _sha256_bytes(state_bytes),
            "size": len(state_bytes),
            "volatile_normalized_path": state_canonical_path.name,
            "volatile_normalized_sha256": _sha256_bytes(state_canonical),
            "volatile_normalized_size": len(state_canonical),
        },
        "amixer": {
            "path": mixer_path.name,
            "sha256": _sha256_bytes(mixer),
            "size": len(mixer),
            "volatile_normalized_path": mixer_canonical_path.name,
            "volatile_normalized_sha256": _sha256_bytes(mixer_canonical),
            "volatile_normalized_size": len(mixer_canonical),
        },
        "volatile_controls": volatile_controls,
        "volatile_allowlist_exact": True,
        "cross_parser_identity_exact": True,
        "normalization_scope": "exact_12_value_lines_only",
        "system_mutation_performed": False,
    }


def _compare_snapshots(before: Mapping[str, Any], after: Mapping[str, Any], generation: Path) -> dict[str, Any]:
    state_before = (generation / str(before["alsactl"]["path"])).read_bytes()
    state_after = (generation / str(after["alsactl"]["path"])).read_bytes()
    mixer_before = (generation / str(before["amixer"]["path"])).read_bytes()
    mixer_after = (generation / str(after["amixer"]["path"])).read_bytes()
    state_canonical_before = (
        generation / str(before["alsactl"]["volatile_normalized_path"])
    ).read_bytes()
    state_canonical_after = (
        generation / str(after["alsactl"]["volatile_normalized_path"])
    ).read_bytes()
    mixer_canonical_before = (
        generation / str(before["amixer"]["volatile_normalized_path"])
    ).read_bytes()
    mixer_canonical_after = (
        generation / str(after["amixer"]["volatile_normalized_path"])
    ).read_bytes()
    if before["volatile_controls"].keys() != after["volatile_controls"].keys():
        raise StateUncertainError("volatile control identity inventory가 전후 다릅니다")
    identity_fields = (
        "numid",
        "name",
        "iface",
        "type",
        "count",
        "alsactl_access",
        "amixer_access",
    )
    for numid in before["volatile_controls"]:
        if any(
            before["volatile_controls"][numid][field]
            != after["volatile_controls"][numid][field]
            for field in identity_fields
        ):
            raise StateUncertainError(f"volatile control {numid} identity/metadata가 전후 다릅니다")
    result = {
        "alsactl_raw_byte_exact": state_before == state_after,
        "alsactl_raw_sha_exact": before["alsactl"]["sha256"] == after["alsactl"]["sha256"],
        "amixer_raw_byte_exact": mixer_before == mixer_after,
        "amixer_raw_sha_exact": before["amixer"]["sha256"] == after["amixer"]["sha256"],
        "alsactl_volatile_normalized_byte_exact": state_canonical_before
        == state_canonical_after,
        "alsactl_volatile_normalized_sha_exact": before["alsactl"][
            "volatile_normalized_sha256"
        ]
        == after["alsactl"]["volatile_normalized_sha256"],
        "amixer_volatile_normalized_byte_exact": mixer_canonical_before
        == mixer_canonical_after,
        "amixer_volatile_normalized_sha_exact": before["amixer"][
            "volatile_normalized_sha256"
        ]
        == after["amixer"]["volatile_normalized_sha256"],
        "volatile_identity_metadata_exact": True,
        "volatile_raw_values_preserved": True,
        "normalization_scope": "exact_12_value_lines_only",
        "automatic_restore_performed": False,
        "state_authority": "nonvolatile_and_volatile_metadata_exact_raw_values_preserved",
    }
    required = (
        "alsactl_volatile_normalized_byte_exact",
        "alsactl_volatile_normalized_sha_exact",
        "amixer_volatile_normalized_byte_exact",
        "amixer_volatile_normalized_sha_exact",
        "volatile_identity_metadata_exact",
    )
    if not all(result[key] is True for key in required):
        raise StateUncertainError(
            "stream 전후 ALSA/mixer nonvolatile bytes 또는 volatile metadata가 다릅니다; "
            "자동 restore하지 않습니다"
        )
    return result


def _sounddevice_fingerprint(module: ModuleType) -> dict[str, Any]:
    path = Path(str(module.__file__)).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("sounddevice module path가 regular file이 아닙니다")
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("sounddevice version을 확인할 수 없습니다")
    portaudio = module.get_portaudio_version()
    return {
        "module_path": str(path),
        "module_file_sha256": _sha256_file(path),
        "version": version,
        "portaudio_version": int(portaudio[0]),
        "portaudio_version_text": str(portaudio[1]),
    }


def _json_device(
    index: int,
    device: Mapping[str, Any],
    *,
    direction: str,
    hostapi_name: str,
) -> dict[str, Any]:
    capability_key = "max_input_channels" if direction == "input" else "max_output_channels"
    return {
        "index": int(index),
        "name": str(device["name"]),
        "hostapi": int(device["hostapi"]),
        "hostapi_name": hostapi_name,
        "max_input_channels": int(device["max_input_channels"]),
        "max_output_channels": int(device["max_output_channels"]),
        "default_samplerate": float(device["default_samplerate"]),
        "direction": direction,
        "required_channels": 2,
        "capability_pass": int(device[capability_key]) >= 2,
    }


def _resolve_exact_portaudio_devices(sd: ModuleType, card_index: int) -> dict[str, Any]:
    devices = list(sd.query_devices())
    resolved: dict[str, Any] = {}
    for direction, pcm, capability in (
        ("input", 1, "max_input_channels"),
        ("output", 0, "max_output_channels"),
    ):
        token = f"hw:{card_index},{pcm}"
        exact_suffix = f"({token})"
        matches = [
            (index, device)
            for index, device in enumerate(devices)
            if str(device["name"]).endswith(exact_suffix)
            and int(device[capability]) >= 2
        ]
        if len(matches) != 1:
            raise RuntimeError(f"PortAudio {direction} {token} mapping이 exact 하나가 아닙니다")
        index, device = matches[0]
        hostapi = sd.query_hostapis(int(device["hostapi"]))
        hostapi_name = str(hostapi["name"])
        if hostapi_name != "ALSA":
            raise RuntimeError(f"PortAudio {direction} hostapi가 ALSA가 아닙니다: {hostapi_name}")
        record = _json_device(
            index,
            device,
            direction=direction,
            hostapi_name=hostapi_name,
        )
        if not record["capability_pass"]:
            raise RuntimeError(f"PortAudio {direction} capability가 부족합니다")
        resolved[direction] = record
    sd.check_input_settings(
        device=resolved["input"]["index"], channels=2, dtype="int32", samplerate=48_000
    )
    sd.check_output_settings(
        device=resolved["output"]["index"], channels=2, dtype="int32", samplerate=48_000
    )
    return resolved


def _parse_hw_params(text: str) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    required = {"format": "S32_LE", "channels": "2", "rate": "48000", "period_size": "256"}
    for key, expected in required.items():
        observed = values.get(key, "")
        if key == "rate":
            observed = observed.split()[0] if observed else ""
        if observed != expected:
            raise RuntimeError(f"hw_params {key}={observed!r}, expected={expected!r}")
    return {
        "format": values["format"],
        "channels": int(values["channels"]),
        "rate": int(values["rate"].split()[0]),
        "period_size": int(values["period_size"]),
        "raw_sha256": _sha256_bytes(text.encode("utf-8")),
    }


def _read_stream_proc(card_index: int) -> dict[str, Any]:
    endpoints = {
        "input": PROC_ASOUND_ROOT / f"card{card_index}/pcm1c/sub0",
        "output": PROC_ASOUND_ROOT / f"card{card_index}/pcm0p/sub0",
    }
    result: dict[str, Any] = {}
    for direction, root in endpoints.items():
        hw_text = _read_text_strict(root / "hw_params")
        status_text = _read_text_strict(root / "status")
        if status_text.splitlines()[0].strip() == "closed":
            raise RuntimeError(f"{direction} stream이 hw_params 검사 전에 닫혔습니다")
        result[direction] = {
            "hw_params": _parse_hw_params(hw_text),
            "status_raw_sha256": _sha256_bytes(status_text.encode("utf-8")),
            "status_first_line": status_text.splitlines()[0].strip(),
        }
    intended = {
        (endpoints["input"] / "status").resolve(strict=True),
        (endpoints["output"] / "status").resolve(strict=True),
    }
    all_statuses = sorted(PROC_ASOUND_ROOT.glob("card*/pcm*/sub*/status"))
    if not all_statuses:
        raise RuntimeError("stream 중 system PCM status inventory가 비었습니다")
    unintended: dict[str, str] = {}
    for path in all_statuses:
        resolved = path.resolve(strict=True)
        first = _read_text_strict(path).splitlines()[0].strip()
        if resolved in intended:
            if first == "closed":
                raise RuntimeError(f"intended stream status가 closed입니다: {path}")
            continue
        unintended[str(path.relative_to(PROC_ASOUND_ROOT))] = first
        if first != "closed":
            raise RuntimeError(f"unintended PCM이 stream 중 열렸습니다: {path} ({first})")
    result["unintended_pcm_diagnostic"] = {
        "all_closed": True,
        "status_count": len(unintended),
        "statuses": unintended,
        "authority": "diagnostic_occupancy_only",
    }
    result["authority"] = {
        "physical_sample_drop_count": None,
        "physical_sample_add_count": None,
        "hardware_deadline_miss_count": None,
        "hardware_sample_slip_authority": False,
        "shared_clock_authority": False,
        "status_semantics": "diagnostic_proc_snapshot_only",
    }
    return result


def _build_transport_receipt(
    *,
    contract: ModuleType,
    plan: Mapping[str, Any],
    planned_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """primitive raw telemetry와 순수 계약 adapter가 같은 receipt를 내는지 대조한다."""

    telemetry_receipt = contract.capture_telemetry_to_contract(
        plan=plan, capture_telemetry=telemetry
    )
    receipt = contract.build_zero_duplex_receipt(
        plan=plan,
        planned_pcm=planned_pcm,
        telemetry=telemetry,
    )
    if receipt.get("telemetry_receipt") != telemetry_receipt:
        raise RuntimeError("primitive adapter/direct contract telemetry receipt가 다릅니다")
    return receipt


def _raw_arrays(captured: np.ndarray, telemetry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = (
        "actual_submitted_pcm",
        "capture_valid_mask",
        "submitted_valid_mask",
        "callback_sequence",
        "callback_start_frames",
        "callback_frame_counts",
        "callback_status_bitmask",
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    )
    arrays = {"captured_pcm": np.ascontiguousarray(captured, dtype="<i4")}
    for name in names:
        if name in telemetry:
            arrays[name] = np.ascontiguousarray(telemetry[name])
    return arrays


def _analyze_simultaneous_input(captured: np.ndarray) -> dict[str, Any]:
    """같은 duplex raw에서 기동 1초를 버리고 두 마이크 생존성을 판정한다."""

    raw = np.asarray(captured)
    settle_frames = 48_000
    if raw.dtype != np.dtype("<i4") or raw.ndim != 2 or raw.shape[1] != 2:
        raise RuntimeError("simultaneous input은 exact <i4 [frames,2]여야 합니다")
    if raw.shape[0] <= settle_frames:
        raise RuntimeError("simultaneous input에 1초 settle 이후 표본이 없습니다")
    audio_io = importlib.import_module("deep_anc.audio_io")
    report = audio_io.analyze_int32_input_probe(
        np.ascontiguousarray(raw[settle_frames:], dtype="<i4"),
        min_rms_dbfs=-80.0,
        max_clip_ratio=0.005,
        min_unique_codes=8,
    )
    channels = report.get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        raise RuntimeError("simultaneous input probe channel report가 불완전합니다")
    if any(channel.get("valid") is not True for channel in channels):
        raise RuntimeError("simultaneous input이 dead/stuck/railed gate를 통과하지 못했습니다")
    return {
        "source": "same_simultaneous_duplex_raw",
        "settle_frames_discarded": settle_frames,
        "settle_seconds_discarded": 1.0,
        "analyzed_frames": int(raw.shape[0] - settle_frames),
        "passed": True,
        "report": report,
    }


def _revalidate_live_static(
    static_public: Mapping[str, Any], *, sounddevice_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """stream 동안 checkout/config/code/backend가 바뀌지 않았음을 postverify한다."""

    git_expected = static_public["git"]
    git_actual = _git_identity(str(git_expected["commit"]))
    if git_actual != git_expected:
        raise RuntimeError("live 중 git identity가 바뀌었습니다")
    config = static_public["config"]
    config_path = _repo_relative_regular(
        REPO_ROOT / str(config["relative_path"]), expected_relative=CONFIG_RELATIVE_PATH
    )
    if _sha256_file(config_path) != config["file_sha256"]:
        raise RuntimeError("live 중 config bytes가 바뀌었습니다")
    script = static_public["script"]
    script_path = _repo_relative_regular(
        REPO_ROOT / str(script["relative_path"]), expected_relative=ADAPTER_RELATIVE_PATH
    )
    if _sha256_file(script_path) != script["file_sha256"]:
        raise RuntimeError("live 중 adapter bytes가 바뀌었습니다")
    expected_python = static_public["python"]
    modules = expected_python["modules"]
    current_python = _assert_current_worktree_binding()
    expected_python_core = {
        key: value for key, value in expected_python.items() if key != "modules"
    }
    if current_python != expected_python_core:
        raise RuntimeError("live 중 Python/venv/package binding이 바뀌었습니다")
    for name, binding in modules.items():
        path = Path(str(binding["path"])).resolve(strict=True)
        if _sha256_file(path) != binding["file_sha256"]:
            raise RuntimeError(f"live 중 {name} module bytes가 바뀌었습니다")
    backend_path = Path(str(sounddevice_binding["module_path"])).resolve(strict=True)
    if _sha256_file(backend_path) != sounddevice_binding["module_file_sha256"]:
        raise RuntimeError("live 중 sounddevice module bytes가 바뀌었습니다")
    tools_after = _tool_fingerprints()
    if tools_after != static_public["tools"]:
        raise RuntimeError("live 중 external tool fingerprint가 바뀌었습니다")
    return {
        "passed": True,
        "git": git_actual,
        "config_file_sha256": config["file_sha256"],
        "script_file_sha256": script["file_sha256"],
        "module_file_sha256": {
            name: binding["file_sha256"] for name, binding in modules.items()
        },
        "sounddevice_file_sha256": sounddevice_binding["module_file_sha256"],
        "tool_fingerprints_exact": True,
    }


def _sounddevice_import() -> ModuleType:
    # 문자열 기반 import는 dry-run/static gate가 backend import를 완전히 피하도록 한다.
    return importlib.import_module("sounddevice")


def _failure_payload(
    *,
    stage: str,
    error: BaseException,
    static: Mapping[str, Any],
    generation: Path,
    raw: Mapping[str, Any] | None,
    state_uncertain: bool,
) -> dict[str, Any]:
    return {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "status": "STATE_UNCERTAIN" if state_uncertain else "FAIL",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "generation": str(generation.relative_to(REPO_ROOT)),
        "static_binding": dict(static),
        "raw": None if raw is None else dict(raw),
        "system_mutation_performed": False,
        "automatic_restore_performed": False,
        "authority": {
            "zero_duplex_transport_smoke_pass": False,
            "hardware_sample_slip_authority": False,
            "shared_clock_authority": False,
            "plant_identification_pass": False,
            "attenuation_assessed": False,
            "canonical_training_eligible": False,
        },
    }


def _static_contract(args: argparse.Namespace, *, live: bool) -> dict[str, Any]:
    config, config_binding = _load_and_validate_config(Path(args.config))
    audio_api, contract_api, python_binding = _import_zero_api()
    smoke = config["zero_duplex_smoke"]
    plan, planned_pcm = contract_api.build_zero_duplex_plan(
        frame_count=int(smoke["expected_frames"])
    )
    if int(plan["callback_count"]) != int(smoke["expected_callbacks"]):
        raise RuntimeError("plan callback count가 config와 다릅니다")
    if getattr(contract_api, "AUTHORITY_CEILING", EXPECTED_AUTHORITY_CEILING) != EXPECTED_AUTHORITY_CEILING:
        raise RuntimeError("contract authority ceiling이 다릅니다")
    git = None
    tools = None
    confirmations = None
    if live:
        _require_live_flags(args)
        confirmations = _physical_confirmations(args)
        tools = _tool_fingerprints()
        git = _git_identity(str(args.expected_commit))
    script_sha = _sha256_file(_repo_relative_regular(SCRIPT_PATH, expected_relative=ADAPTER_RELATIVE_PATH))
    return {
        "config": config,
        "config_binding": config_binding,
        "audio_api": audio_api,
        "contract_api": contract_api,
        "python_binding": python_binding,
        "plan": plan,
        "planned_pcm": planned_pcm,
        "git": git,
        "tools": tools,
        "physical_confirmations": confirmations,
        "script": {"relative_path": ADAPTER_RELATIVE_PATH, "file_sha256": script_sha},
        "generation": REPO_ROOT / str(smoke["result_directory"]),
    }


def _execute_live(args: argparse.Namespace) -> int:
    generation: Path | None = None
    generation_claimed = False
    static_public: dict[str, Any] = {}
    raw_binding: dict[str, Any] | None = None
    pending_receipt: dict[str, Any] | None = None
    stage = "static_preflight"
    try:
        static = _static_contract(args, live=True)
        generation = static["generation"]
        static_public = {
            "config": static["config_binding"],
            "python": static["python_binding"],
            "git": static["git"],
            "script": static["script"],
            "tools": static["tools"],
            "physical_confirmations": static["physical_confirmations"],
            "plan_payload_sha256": static["plan"]["canonical_payload_sha256"],
            "planned_pcm_sha256": static["plan"]["planned_pcm_sha256"],
        }
        claim = {
            "schema": CLAIM_SCHEMA,
            "status": "CONSUMED_NO_RERUN",
            "generation": str(generation.relative_to(REPO_ROOT)),
            "static_binding": static_public,
            "created_time_ns": time.time_ns(),
        }
        stage = "generation_claim"
        claim_binding = _claim_generation(generation, claim)
        generation_claimed = True
        _write_json_exclusive(generation / "sealed_plan.json", static["plan"])

        with (
            _machine_global_audio_lock() as machine_lock,
            _repository_live_audio_lock() as repository_lock,
            _DeferredSignalScope() as signals,
        ):
            stage = "read_only_precheck"
            preconditions = assert_live_pcm_clock_preconditions()
            card_index = int(preconditions["alsa_card_index"])
            physical_fingerprint = _collect_alsa_physical_fingerprint(static["config"])
            before = _snapshot_read_only(generation, label="before")
            snapshot_marker = {
                "schema": SNAPSHOT_SCHEMA,
                "status": "READ_ONLY_BASELINE_DURABLE",
                "before": before,
                "system_mutation_performed": False,
                "restore_command_permitted": False,
            }
            _write_json_exclusive(generation / "read_only_baseline.json", snapshot_marker)
            signals.raise_if_pending()

            stage = "backend_binding"
            sd = _sounddevice_import()
            backend_binding = _sounddevice_fingerprint(sd)
            devices = _resolve_exact_portaudio_devices(sd, card_index)
            # query/check API 이후에도 다른 process가 PCM을 열 수 있으므로 Stream 직전 재검증한다.
            assert_live_pcm_clock_preconditions(expected_card_index=card_index)
            signals.raise_if_pending()

            stream_proc: dict[str, Any] = {}

            def pre_open_check() -> None:
                assert_live_pcm_clock_preconditions(expected_card_index=card_index)

            def on_stream_started() -> None:
                stream_proc.update(_read_stream_proc(card_index))

            stage = "zero_duplex_stream"
            captured: np.ndarray
            telemetry: dict[str, Any]
            capture_error: BaseException | None = None
            try:
                captured, telemetry = static["audio_api"].capture_zero_duplex(
                    sd,
                    total_frames=int(static["plan"]["frame_count"]),
                    input_device=int(devices["input"]["index"]),
                    output_device=int(devices["output"]["index"]),
                    pre_open_check=pre_open_check,
                    on_stream_started=on_stream_started,
                    watchdog_grace_seconds=float(static["plan"]["watchdog_grace_seconds"]),
                )
            except static["audio_api"].ZeroDuplexCaptureFailure as error:
                capture_error = error
                captured = np.ascontiguousarray(error.captured_pcm, dtype="<i4")
                telemetry = dict(error.telemetry)
                telemetry["actual_submitted_pcm"] = np.ascontiguousarray(
                    error.actual_submitted_pcm, dtype="<i4"
                )
                telemetry["capture_valid_mask"] = np.ascontiguousarray(error.capture_valid_mask)
                telemetry["submitted_valid_mask"] = np.ascontiguousarray(error.submitted_valid_mask)

            # capture 함수 반환/예외는 Stream close 경로가 끝났다는 뜻이다. 유일한 actual
            # raw를 post snapshot보다 먼저 보존해 뒤 단계 실패가 raw까지 잃게 하지 않는다.
            stage = "raw_publication"
            arrays = _raw_arrays(captured, telemetry)
            raw_binding = _publish_npz_exclusive(generation / "raw_capture.npz", arrays)

            stage = "read_only_postverify"
            try:
                after = _snapshot_read_only(generation, label="after")
                comparison = _compare_snapshots(before, after, generation)
                postconditions = assert_live_pcm_clock_preconditions(
                    expected_card_index=card_index
                )
            except StateUncertainError:
                raise
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise StateUncertainError(
                    f"post-stream read-only state를 exact 검증하지 못했습니다: {error}"
                ) from error
            signals.raise_if_pending()

            stage = "static_post_revalidation"
            static_post_revalidation = _revalidate_live_static(
                static_public, sounddevice_binding=backend_binding
            )
            if capture_error is not None:
                raise capture_error

            stage = "simultaneous_input_probe"
            input_probe = _analyze_simultaneous_input(captured)

            stage = "contract_validation"
            transport_receipt = _build_transport_receipt(
                contract=static["contract_api"],
                plan=static["plan"],
                planned_pcm=static["planned_pcm"],
                telemetry=telemetry,
            )
            pending_receipt = {
                "schema": LIVE_RECEIPT_SCHEMA,
                "status": EXPECTED_AUTHORITY_CEILING,
                "valid": True,
                "generation": str(generation.relative_to(REPO_ROOT)),
                "generation_claim": claim_binding,
                "static_binding": static_public,
                "machine_global_lock": machine_lock,
                "repository_audio_lock": repository_lock,
                "preconditions": preconditions,
                "physical_fingerprint": physical_fingerprint,
                "sounddevice": backend_binding,
                "resolved_devices": devices,
                "simultaneous_input_probe": input_probe,
                "static_post_revalidation": static_post_revalidation,
                "stream_proc_diagnostic": stream_proc,
                "stream_close_semantics": "capture_returned_after_stream_close_without_exception",
                "postconditions": postconditions,
                "read_only_snapshots": {
                    "before": before,
                    "after": after,
                    "comparison": comparison,
                    "system_mutation_performed": False,
                    "automatic_restore_performed": False,
                },
                "raw": raw_binding,
                "transport_contract_receipt": transport_receipt,
                "physical_counters": {
                    "drop_sample_count": None,
                    "add_sample_count": None,
                    "deadline_miss_count": None,
                    "hardware_sample_slip_authority": False,
                },
                "authority": {
                    "zero_duplex_transport_smoke_pass": True,
                    "common_clock_topology_pass": False,
                    "shared_clock_authority_pass": False,
                    "hardware_sample_slip_authority": False,
                    "sample_identity_pass": False,
                    "physical_output_route_pass": False,
                    "plant_identification_pass": False,
                    "attenuation_assessed": False,
                    "canonical_training_eligible": False,
                },
            }
            # input/contract 계산 중 들어온 signal도 lock/handler cleanup 전에 실패로 승격한다.
            signals.raise_if_pending()

        if pending_receipt is None:
            raise RuntimeError("lock/signal scope 종료 뒤 success receipt payload가 없습니다")
        stage = "success_publication"
        if (generation / "failure.json").exists():
            raise RuntimeError("failure receipt와 success receipt를 함께 발행할 수 없습니다")
        pending_receipt["transaction_finalization"] = {
            "signal_handlers_restored_before_success_publication": True,
            "repository_audio_lock_released_before_success_publication": True,
            "machine_global_lock_released_before_success_publication": True,
        }
        _publish_terminal_receipt(generation, success=True, value=pending_receipt)
        print(
            "[PASS] exact-zero duplex transport smoke만 통과했습니다. "
            "shared clock/sample identity/P/S/ANC 권한은 없습니다."
        )
        print(f"[보존] {generation / 'raw_capture.npz'}")
        print(f"[보존] {generation / 'receipt.json'}")
        return 0
    except BaseException as error:
        success_exists = bool(
            generation is not None and (generation / "receipt.json").exists()
        )
        if (
            generation_claimed
            and not success_exists
            and generation is not None
            and generation.is_dir()
        ):
            failure = _failure_payload(
                stage=stage,
                error=error,
                static=static_public,
                generation=generation,
                raw=raw_binding,
                state_uncertain=isinstance(error, StateUncertainError),
            )
            try:
                if (generation / "failure.json").exists():
                    raise FileExistsError("failure receipt가 이미 존재합니다")
                _publish_terminal_receipt(generation, success=False, value=failure)
            except BaseException as publish_error:
                print(f"[보존 실패] failure receipt: {publish_error}", file=sys.stderr)
        elif success_exists:
            print(
                "[보존 경고] success receipt가 이미 존재하므로 failure와 공존시키지 않았습니다.",
                file=sys.stderr,
            )
        print(f"[실패] {stage}: {type(error).__name__}: {error}", file=sys.stderr)
        if isinstance(error, StateUncertainError):
            print(
                "[STATE_UNCERTAIN] 외부 ALSA/mixer 변화 가능성 때문에 자동 restore하지 않았습니다.",
                file=sys.stderr,
            )
        return 1


def _dry_run(args: argparse.Namespace) -> int:
    static = _static_contract(args, live=False)
    generation = static["generation"]
    print("[DRY-RUN PASS] sounddevice import/장치 open/system mutation 없음")
    print(f"config SHA256: {static['config_binding']['file_sha256']}")
    print(f"script SHA256: {static['script']['file_sha256']}")
    print(f"plan SHA256: {static['plan']['canonical_payload_sha256']}")
    print("예상 stream 시간: 60.000초 | 의도된 audible signal: 0초 (exact zero)")
    print("주의: exact zero여도 codec/stream open·close pop 가능성이 있어 물리 분리가 필수입니다.")
    print(f"generation(no-replace, 한 번만): {generation.relative_to(REPO_ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="장치를 열지 않는 기본 검증")
    mode.add_argument("--execute-live", action="store_true", help="exact-zero duplex 60초를 한 번 실행")
    parser.add_argument("--config", default=str(REPO_ROOT / CONFIG_RELATIVE_PATH))
    parser.add_argument("--expected-commit")
    parser.add_argument("--confirm-j511-disconnected", action="store_true")
    parser.add_argument("--confirm-amplifier-power-off", action="store_true")
    parser.add_argument("--confirm-amplifier-input-disconnected", action="store_true")
    parser.add_argument("--confirm-ab13x-amplifier-disconnected", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_live:
        try:
            return _dry_run(args)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"[DRY-RUN FAIL] {type(error).__name__}: {error}", file=sys.stderr)
            return 2
    return _execute_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
