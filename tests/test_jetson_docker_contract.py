"""Jetson Docker 이미지의 설치·실행 계약 회귀 테스트."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRELOAD_SCRIPT = REPO_ROOT / "docker/preload_jetson.py"
DOCKERFILE = REPO_ROOT / "docker/Dockerfile.jetson"
DEV_SCRIPT = REPO_ROOT / "scripts/docker/dev.sh"


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


def test_dockerfile_avoids_cross_layer_venv_copy_up():
    text = DOCKERFILE.read_text(encoding="utf-8")

    install = text.index("RUN python3 -m venv")
    venv_chown = text.index("chown -R ${DEV_UID}:${DEV_GID} .venv")
    source_copy = text.index("COPY --chown=${DEV_UID}:${DEV_GID} src/ src/")
    non_root = text.index("USER ancdev")
    editable = text.index("RUN .venv/bin/python -m pip install -e .")

    assert install < venv_chown < source_copy < non_root < editable
    assert "chown -R ${DEV_UID}:${DEV_GID} /workspace" not in text
    assert "'nvidia-cuda-cupti-cu12==12.6.68'" in text
    assert "'nvidia-cusparselt-cu12==0.6.2'" in text
    assert "nvidia-cuda-cupti-cu12 nvidia-cusparselt-cu12" not in text
    assert "Path('/.dockerenv')" not in PRELOAD_SCRIPT.read_text(encoding="utf-8")


def test_jetson_commands_bypass_broken_default_bridge_without_exposing_audio():
    text = DEV_SCRIPT.read_text(encoding="utf-8")

    assert 'task_build_extra+=(--network host)' in text
    assert 'docker build "${task_build_extra[@]}"' in text
    assert 'task_extra+=(--runtime nvidia --network host)' in text
    assert "/dev/snd" not in text
