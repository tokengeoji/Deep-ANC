"""Jetson Docker 이미지의 설치·실행 계약 회귀 테스트."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRELOAD_SCRIPT = REPO_ROOT / "docker/preload_jetson.py"
DOCKERFILE = REPO_ROOT / "docker/Dockerfile.jetson"
DEV_SCRIPT = REPO_ROOT / "scripts/docker/dev.sh"
LOCAL_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.jetson-local"
LOCAL_BOOTSTRAP = REPO_ROOT / "scripts/docker/bootstrap_jetson_local.sh"


def _load_preload_module():
    spec = importlib.util.spec_from_file_location("deep_anc_preload_jetson", PRELOAD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("in_venv", "container_marker"),
    ((False, "jetson"), (True, None), (True, "cpu")),
)
def test_preload_rejects_invalid_install_context(
    monkeypatch: pytest.MonkeyPatch,
    in_venv: bool,
    container_marker: str | None,
):
    module = _load_preload_module()
    monkeypatch.setattr(module.sys, "prefix", "/venv" if in_venv else "/usr")
    monkeypatch.setattr(module.sys, "base_prefix", "/usr")
    if container_marker is None:
        monkeypatch.delenv("DEEP_ANC_CONTAINER", raising=False)
    else:
        monkeypatch.setenv("DEEP_ANC_CONTAINER", container_marker)

    with pytest.raises(SystemExit, match="Docker 컨테이너의 venv"):
        module.main()


def test_preload_writes_hooks_only_to_selected_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_preload_module()
    monkeypatch.setattr(module.sys, "prefix", "/venv")
    monkeypatch.setattr(module.sys, "base_prefix", "/usr")
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "jetson")
    monkeypatch.setattr(module.sysconfig, "get_path", lambda name: str(tmp_path))

    module.main()

    hook = (tmp_path / "_deep_anc_libpaths.py").read_text(encoding="utf-8")
    pth = (tmp_path / "_deep_anc_libs.pth").read_text(encoding="utf-8")
    assert "nvtx/lib/libnvToolsExt.so.1" in hook
    assert "cuda_cupti/lib/libcupti.so.12" in hook
    assert "cusparselt/lib/libcusparseLt.so.0" in hook
    assert pth == "import _deep_anc_libpaths\n"


def test_generated_preload_loads_top_level_cusparselt_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_preload_module()
    monkeypatch.setattr(module.sys, "prefix", "/venv")
    monkeypatch.setattr(module.sys, "base_prefix", "/usr")
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "jetson")
    monkeypatch.setattr(module.sysconfig, "get_path", lambda name: str(tmp_path))
    relatives = (
        "nvidia/nvtx/lib/libnvToolsExt.so.1",
        "nvidia/cuda_cupti/lib/libcupti.so.12",
        "cusparselt/lib/libcusparseLt.so.0",
    )
    # 0.6.2 실제 배치와 잘못 가정했던 nvidia/ 배치를 함께 두어 선택을 검증한다.
    for relative in (*relatives, "nvidia/cusparselt/lib/libcusparseLt.so.0"):
        library = tmp_path / relative
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    loaded = []
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path, mode: loaded.append((Path(path), mode))
    )
    module.main()

    hook_path = tmp_path / "_deep_anc_libpaths.py"
    spec = importlib.util.spec_from_file_location("generated_jetson_preload", hook_path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert loaded == [(tmp_path / relative, os.RTLD_GLOBAL) for relative in relatives]


def test_dockerfile_avoids_cross_layer_venv_copy_up():
    text = DOCKERFILE.read_text(encoding="utf-8")

    install = text.index("RUN python3 -m venv")
    venv_chown = text.index("chown -R ${DEV_UID}:${DEV_GID} .venv")
    source_copy = text.index("COPY --chown=${DEV_UID}:${DEV_GID} src/ src/")
    non_root = text.index("USER ancdev")
    editable = text.index("RUN .venv/bin/python -m pip install -e .")

    assert install < venv_chown < source_copy < non_root < editable
    assert "chown -R ${DEV_UID}:${DEV_GID} /workspace" not in text
    requirements = (REPO_ROOT / "docker/requirements-jetson-runtime.txt").read_text(encoding="utf-8")
    assert "nvidia-cuda-cupti-cu12==12.6.68" in requirements
    assert "nvidia-cusparselt-cu12==0.6.2" in requirements
    assert "-r docker/requirements-jetson-runtime.txt" in text
    assert "Path('/.dockerenv')" not in PRELOAD_SCRIPT.read_text(encoding="utf-8")


def test_jetson_commands_bypass_broken_default_bridge_without_exposing_audio():
    text = DEV_SCRIPT.read_text(encoding="utf-8")

    assert 'task_build_extra+=(--network host)' in text
    assert 'docker build "${task_build_extra[@]}"' in text
    assert 'task_extra+=(--runtime nvidia --network host)' in text
    assert "/dev/snd" not in text


@pytest.fixture
def shell_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """호스트 Docker와 실제 venv를 호출하지 않는 셸 실행 경계."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stubs = {
        "docker": 'printf "%s\\n" "$@" > "$DOCKER_CALL_LOG"\n',
        "uname": 'printf "%s\\n" "$TEST_ARCH"\n',
        "mountpoint": 'printf "mountpoint\\n" >> "$BOOTSTRAP_TRACE"\nexit 1\n',
        "flock": 'printf "flock\\n" >> "$BOOTSTRAP_TRACE"\nexit 91\n',
        "python3": 'printf "python3\\n" >> "$BOOTSTRAP_TRACE"\nexit 92\n',
    }
    for name, body in stubs.items():
        executable = stub_dir / name
        executable.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("DOCKER_CALL_LOG", str(tmp_path / "docker-call.txt"))
    monkeypatch.setenv("BOOTSTRAP_TRACE", str(tmp_path / "bootstrap-trace.txt"))
    monkeypatch.setenv("TEST_ARCH", "aarch64")
    monkeypatch.delenv("BASH_ENV", raising=False)
    return tmp_path


def _run_shell(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("target", "architecture"),
    (("cpu", "x86_64"), ("jetson", "aarch64"), ("jetson-local", "aarch64")),
)
def test_build_preserves_sudo_user_and_target_arguments(
    shell_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    architecture: str,
):
    monkeypatch.setenv("TEST_ARCH", architecture)
    monkeypatch.setenv("SUDO_UID", "2468")
    monkeypatch.setenv("SUDO_GID", "1357")

    result = _run_shell(DEV_SCRIPT, "build", target)

    assert result.returncode == 0, result.stderr
    arguments = (shell_sandbox / "docker-call.txt").read_text(encoding="utf-8").splitlines()
    network = ["--network", "host"] if target.startswith("jetson") else []
    assert arguments == [
        "build", *network,
        "--build-arg", "DEV_UID=2468", "--build-arg", "DEV_GID=1357",
        "-f", str(REPO_ROOT / f"docker/Dockerfile.{target}"),
        "-t", f"deep-anc-{target}:dev", str(REPO_ROOT),
    ]


@pytest.mark.parametrize("action", ("build", "up"))
@pytest.mark.parametrize(
    ("target", "architecture", "status", "message"),
    (
        ("jetson-extra", "aarch64", 2, "사용:"),
        ("cpu", "aarch64", 1, "x86_64용"),
        ("jetson", "x86_64", 1, "ARM64 Jetson"),
        ("jetson-local", "x86_64", 1, "ARM64 Jetson"),
    ),
)
def test_invalid_target_or_architecture_fails_before_docker(
    shell_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    target: str,
    architecture: str,
    status: int,
    message: str,
):
    monkeypatch.setenv("TEST_ARCH", architecture)

    result = _run_shell(DEV_SCRIPT, action, target)

    assert result.returncode == status
    assert message in result.stderr
    assert not (shell_sandbox / "docker-call.txt").exists()


@pytest.mark.parametrize(
    ("marker", "message", "trace"),
    (
        ("cpu", "실제 Jetson Docker", ""),
        ("jetson", "별도 Docker 볼륨", "mountpoint\n"),
    ),
)
def test_local_bootstrap_rejects_context_before_installing(
    shell_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    message: str,
    trace: str,
):
    if marker == "jetson" and not Path("/.dockerenv").is_file():
        pytest.skip("별도 venv 볼륨 게이트 검사는 Docker 내부에서 실행한다.")
    # 절대 작업 경로만 임시 폴더로 연결한다. 회귀가 생겨도 실제 venv와 pip에는 닿지 않는다.
    bash_env = shell_sandbox / "bootstrap-env.sh"
    bash_env.write_text(
        'cd() {\n'
        '  if [[ "${1-}" == /workspace/Deep-ANC ]]; then\n'
        '    builtin cd "$BOOTSTRAP_SANDBOX"\n'
        '  else builtin cd "$@"; fi\n'
        '}\n',
        encoding="utf-8",
    )
    venv_bin = shell_sandbox / ".venv/bin"
    venv_bin.mkdir(parents=True)
    python_stub = venv_bin / "python"
    python_stub.write_text(
        '#!/bin/sh\nprintf "venv-python\\n" >> "$BOOTSTRAP_TRACE"\nexit 93\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    monkeypatch.setenv("BASH_ENV", str(bash_env))
    monkeypatch.setenv("BOOTSTRAP_SANDBOX", str(shell_sandbox))
    monkeypatch.setenv("DEEP_ANC_CONTAINER", marker)

    result = _run_shell(LOCAL_BOOTSTRAP)

    assert result.returncode == 1, result.stderr
    assert message in result.stderr
    trace_path = shell_sandbox / "bootstrap-trace.txt"
    assert (trace_path.read_text(encoding="utf-8") if trace_path.exists() else "") == trace
    assert not (shell_sandbox / ".venv/.bootstrap.lock").exists()


def test_local_runtime_mounts_only_readonly_nvidia_libraries_without_audio():
    text = DEV_SCRIPT.read_text(encoding="utf-8")

    assert "task_lib_root=/usr/lib/aarch64-linux-gnu" in text
    assert '"$task_lib_root"/libcudnn*.so.9*' in text
    assert '"$task_lib_root"/libnvinfer*.so.10*' in text
    assert '"$task_lib_root"/libnvonnxparser*.so.10*' in text
    assert '"type=bind,src=$task_lib,dst=$task_lib,readonly"' in text
    assert "task_trt=/usr/lib/python3.10/dist-packages/tensorrt" in text
    assert '"type=bind,src=$task_trt,dst=$task_trt,readonly"' in text
    assert "type=bind,src=$task_lib_root," not in text
    assert "/dev/snd" not in text
    assert "--privileged" not in text
    assert "--device" not in text


def test_local_bootstrap_shares_pinned_runtime_requirements_without_image_copy():
    bootstrap = LOCAL_BOOTSTRAP.read_text(encoding="utf-8")
    full_image = DOCKERFILE.read_text(encoding="utf-8")
    local_image = LOCAL_DOCKERFILE.read_text(encoding="utf-8")

    for installer in (bootstrap, full_image):
        assert "-r requirements-jetson.txt -r docker/requirements-dev.txt" in installer
        assert "-r docker/requirements-jetson-runtime.txt" in installer
        assert "docker/preload_jetson.py" in installer
        assert "nv24.08.17622132" not in installer
    assert "pip install" not in local_image
    assert "python3 -m venv" not in local_image
    assert "install -d -o ${DEV_UID} -g ${DEV_GID} /workspace/Deep-ANC/.venv" in local_image
