"""무오디오 스택 진단의 비교·환경 거부·출력 보존 계약 (GPU 연산은 실행하지 않음)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/bench/check_jetson_stack.py"
    spec = importlib.util.spec_from_file_location("jetson_stack_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_reports_measured_error_and_accepts_small_roundoff(diagnostic):
    reference = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    actual = reference.copy()
    actual[2] = 5e-6

    result = diagnostic._compare(np, reference, actual)

    assert result["passed"] is True
    assert result["shape"] == [3]
    assert result["max_abs_error"] == pytest.approx(5e-6)
    assert result["mean_abs_error"] == pytest.approx(5e-6 / 3)


def test_compare_rejects_shape_mismatch_before_broadcast(diagnostic):
    with pytest.raises(RuntimeError, match="shape 불일치"):
        diagnostic._compare(np, np.zeros((2, 1)), np.zeros((2,)))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_reference", [False, True])
def test_compare_rejects_nonfinite_on_either_side(diagnostic, invalid, invalid_reference):
    finite = np.zeros(2)
    bad = np.array([0.0, invalid])
    reference, actual = (bad, finite) if invalid_reference else (finite, bad)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        diagnostic._compare(np, reference, actual)


def test_failed_comparison_keeps_error_in_partial_report(diagnostic):
    result = diagnostic._compare(np, np.zeros(2), np.ones(2))
    report = {"checks": {}}

    with pytest.raises(RuntimeError, match="허용 범위"):
        diagnostic._require_comparison(report, "numeric_check", result)

    assert report["checks"]["numeric_check"]["passed"] is False
    assert report["checks"]["numeric_check"]["max_abs_error"] == 1.0


def test_non_docker_guard_runs_before_backend_imports(diagnostic, monkeypatch):
    paths = []

    def missing_marker(path):
        paths.append(path)
        return SimpleNamespace(exists=lambda: False)

    monkeypatch.setattr(diagnostic, "Path", missing_marker)
    with pytest.raises(RuntimeError, match="Docker 컨테이너 내부"):
        diagnostic._check_stack({})
    assert paths == ["/.dockerenv"]


def test_non_arm_guard_runs_before_l4t_or_backend_access(diagnostic, monkeypatch):
    paths = []

    def docker_marker(path):
        paths.append(path)
        return SimpleNamespace(exists=lambda: True)

    monkeypatch.setattr(diagnostic, "Path", docker_marker)
    monkeypatch.setattr(diagnostic.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="ARM64 Jetson"):
        diagnostic._check_stack({})
    assert paths == ["/.dockerenv"]


def test_main_writes_guard_failure_as_json_without_loading_gpu(
    diagnostic, monkeypatch, tmp_path, capsys,
):
    target = tmp_path / "diagnostic.json"
    monkeypatch.setattr(diagnostic.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diagnostic.sys, "argv", ["check_stack", "--output", str(target)])

    assert diagnostic.main() == 1

    stdout = capsys.readouterr().out
    report = json.loads(stdout)
    assert target.read_text(encoding="utf-8") == stdout
    assert report["status"] == "failed"
    assert report["error"]["type"] == "RuntimeError"
    assert report["checks"] == {}
    assert report["versions"] == {}
    assert "ANC 감쇠" in report["not_validated"]


def test_existing_output_is_untouched_and_stack_is_not_started(
    diagnostic, monkeypatch, tmp_path, capsys,
):
    target = tmp_path / "existing.json"
    target.write_text("기존 진단 기록\n", encoding="utf-8")
    monkeypatch.setattr(diagnostic.sys, "argv", ["check_stack", "--output", str(target)])

    def unexpected_start(report):
        pytest.fail("기존 출력 경로가 있으면 환경 검사도 시작하면 안 됩니다")

    monkeypatch.setattr(diagnostic, "_check_stack", unexpected_start)
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()

    assert exc.value.code == 2
    assert target.read_text(encoding="utf-8") == "기존 진단 기록\n"
    assert "덮어쓰지 않습니다" in capsys.readouterr().err


def test_exclusive_open_preserves_file_even_if_existence_check_misses_it(
    diagnostic, monkeypatch, tmp_path, capsys,
):
    target = tmp_path / "racing_output.json"
    target.write_text("보존할 내용\n", encoding="utf-8")
    original_exists = Path.exists

    def missed_target(path):
        return False if path == target else original_exists(path)

    monkeypatch.setattr(Path, "exists", missed_target)
    monkeypatch.setattr(diagnostic.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diagnostic.sys, "argv", ["check_stack", "--output", str(target)])

    assert diagnostic.main() == 1

    captured = capsys.readouterr()
    assert target.read_text(encoding="utf-8") == "보존할 내용\n"
    assert "JSON 기록 실패" in captured.err
    assert json.loads(captured.out)["status"] == "failed"
