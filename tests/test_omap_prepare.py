"""OMAP 오프라인 준비는 Docker 경계와 기존 의존성 계약을 유지한다."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepare_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """실제 Docker/Python/venv에 닿지 않는 임시 저장소와 실행 파일들."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "scripts/docker").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    source = (ROOT / "tools/prepare_jetson.sh").read_text(encoding="utf-8")
    # 실제 테스트는 Docker에서 수행한다. marker 파일 경로만 바꿔 호스트도 재현한다.
    assert source.count("/.dockerenv") == 1
    source = source.replace("/.dockerenv", str(tmp_path / "container-marker"))
    (tmp_path / "tools/prepare_jetson.sh").write_text(source, encoding="utf-8")
    (tmp_path / "scripts/docker/dev.sh").write_text(
        (ROOT / "scripts/docker/dev.sh").read_text(encoding="utf-8"), encoding="utf-8"
    )
    docker = tmp_path / "bin/docker"
    docker.write_text(
        '#!/bin/sh\n'
        'printf "%s\\0" "$@" >> "$PREPARE_DOCKER_LOG"\n'
        'printf "\\n" >> "$PREPARE_DOCKER_LOG"\n'
        'case "$1" in\n'
        '  container) printf "%s\\n" "$MOCK_DOCKER_RUNNING"; exit "$MOCK_DOCKER_INSPECT_STATUS" ;;\n'
        '  exec) exit 0 ;;\n'
        '  *) exit 95 ;;\n'
        'esac\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    _python_stub(tmp_path / ".venv/bin/python")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("PREPARE_DOCKER_LOG", str(tmp_path / "docker.log"))
    monkeypatch.setenv("PREPARE_PYTHON_LOG", str(tmp_path / "python.log"))
    monkeypatch.setenv("MOCK_DOCKER_RUNNING", "true")
    monkeypatch.setenv("MOCK_DOCKER_INSPECT_STATUS", "0")
    for key in ("DEEP_ANC_CONTAINER", "PYTHON_BIN", "BASH_ENV"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def _python_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/bin/sh\n'
        'printf "%s\\0" "$0" "$@" >> "$PREPARE_PYTHON_LOG"\n'
        'printf "\\n" >> "$PREPARE_PYTHON_LOG"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [line.rstrip(b"\0").decode().split("\0") for line in path.read_bytes().splitlines()]


def _run(sandbox: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(sandbox / "tools/prepare_jetson.sh"), *arguments],
        cwd=sandbox.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_host_routes_unchanged_arguments_through_existing_docker(prepare_sandbox: Path):
    args = ["--allow-cpu", "--install", "--python", "/container/path with spaces/python"]
    result = _run(prepare_sandbox, *args)

    assert result.returncode == 0, result.stderr
    assert _calls(prepare_sandbox / "docker.log") == [
        ["container", "inspect", "--format", "{{.State.Running}}", "deep-anc-dev"],
        ["exec", "-i", "deep-anc-dev", "bash", "tools/prepare_jetson.sh", *args],
    ]
    assert not (prepare_sandbox / "python.log").exists()
    assert not (prepare_sandbox / "runs").exists()


@pytest.mark.parametrize(("running", "status"), [("false", "0"), ("", "1")])
def test_host_requires_running_environment_before_any_python(
    prepare_sandbox: Path, monkeypatch: pytest.MonkeyPatch, running: str, status: str
):
    monkeypatch.setenv("MOCK_DOCKER_RUNNING", running)
    monkeypatch.setenv("MOCK_DOCKER_INSPECT_STATUS", status)
    result = _run(prepare_sandbox, "--install", "--allow-cpu")

    assert result.returncode == 1
    assert "scripts/docker/dev.sh start" in result.stderr
    assert "scripts/docker/dev.sh build cpu" in result.stderr
    assert "scripts/docker/dev.sh up jetson-local" in result.stderr
    assert len(_calls(prepare_sandbox / "docker.log")) == 1
    assert not (prepare_sandbox / "python.log").exists()


def test_cpu_container_preserves_packages_and_creates_fresh_smoke_outputs(
    prepare_sandbox: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    first = _run(prepare_sandbox, "--install", "--allow-cpu")
    second = _run(prepare_sandbox, "--allow-cpu")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert "패키지를 설치하지 않고" in first.stdout
    assert "CPU Docker" in first.stdout
    calls = _calls(prepare_sandbox / "python.log")
    assert len(calls) == 6
    assert all(call[0] == str(prepare_sandbox / ".venv/bin/python") for call in calls)
    assert calls[0][1:] == [str(prepare_sandbox / "tools/jetson_preflight.py"), "--allow-cpu"]
    assert calls[1][1:] == [
        str(prepare_sandbox / "tools/prepare_secondary_path.py"),
        "--output-dir", str(prepare_sandbox / "artifacts/secondary_path"),
    ]
    smoke_calls = [calls[2], calls[5]]
    for call in smoke_calls:
        assert call[1:3] == ["-m", "deepanc.train"]
        assert call[call.index("--device") + 1] == "cpu"
        assert Path(call[call.index("--output") + 1]).is_dir()
    assert smoke_calls[0][-1] != smoke_calls[1][-1]
    assert not any("pip" in call or "venv" in call for call in calls)
    assert not (prepare_sandbox / "docker.log").exists()


def test_dockerenv_marker_is_sufficient_and_default_requires_jetson_cuda(prepare_sandbox: Path):
    (prepare_sandbox / "container-marker").touch()
    result = _run(prepare_sandbox)

    assert result.returncode == 0, result.stderr
    calls = _calls(prepare_sandbox / "python.log")
    assert calls[0][-2:] == ["--require-jetson", "--require-cuda"]
    assert calls[2][calls[2].index("--device") + 1] == "cuda"
    assert not (prepare_sandbox / "docker.log").exists()


def test_container_honors_explicit_python_path_with_spaces(
    prepare_sandbox: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    selected = prepare_sandbox / "custom env/python"
    _python_stub(selected)
    result = _run(prepare_sandbox, "--allow-cpu", "--python", str(selected))

    assert result.returncode == 0, result.stderr
    assert all(call[0] == str(selected) for call in _calls(prepare_sandbox / "python.log"))


def test_missing_container_python_does_not_fall_back_or_install(
    prepare_sandbox: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    (prepare_sandbox / ".venv/bin/python").unlink()
    result = _run(prepare_sandbox, "--install", "--allow-cpu")

    assert result.returncode == 1
    assert "docker/README.md" in result.stderr
    assert not (prepare_sandbox / "python.log").exists()
    assert not (prepare_sandbox / "docker.log").exists()


@pytest.mark.parametrize("arguments", [("--python",), ("--unknown",)])
def test_invalid_arguments_fail_without_launching_anything(prepare_sandbox: Path, arguments: tuple[str, ...]):
    result = _run(prepare_sandbox, *arguments)

    assert result.returncode == 2
    assert not (prepare_sandbox / "python.log").exists()
    assert not (prepare_sandbox / "docker.log").exists()
