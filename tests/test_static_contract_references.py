from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import conftest as repository_conftest
from scripts.ci.check_static_contract_references import (
    FAILURE_PREFIX,
    PASS_PREFIX,
    StaticContractReferenceError,
    audit_static_contract_references,
)


REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "ci" / "check_static_contract_references.py"
CONFTEST = REPO / "tests" / "conftest.py"

V1_BUILDER = "7c7800fa94a8c5e156e049be896fd0b9586d983f"
V2_BUILDER = "0cb13b14e36c334783953aedd47aa0bc13d0fb6a"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_repo(root: Path, *, registry: str | None = None) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    _write(
        root / "src" / "registry.py",
        registry
        or 'NODE = "tests/test_contract.py::test_boundary[param-1]"\n',
    )
    _write(root / "tests" / "test_contract.py", "def test_boundary():\n    pass\n")


def test_static_api_resolves_literals_fstrings_and_parameter_suffixes(tmp_path: Path) -> None:
    _valid_repo(
        tmp_path,
        registry=(
            '_TARGET = "tests/test_contract.py"\n'
            'NEGATIVE = f"{_TARGET}::test_boundary[param-1]"\n'
            'POSITIVE = "tests/test_contract.py::" + "test_other"\n'
        ),
    )
    _write(
        tmp_path / "tests" / "test_contract.py",
        "def test_boundary():\n    pass\n\ndef test_other():\n    pass\n",
    )

    result = audit_static_contract_references(tmp_path)

    assert result.references == 2
    assert result.test_files == 1


@pytest.mark.parametrize(
    ("node_id", "message"),
    [
        ("tests/test_contract.py:test_boundary", "exactly one '::'"),
        ("tests/../test_contract.py::test_boundary", "must stay inside the repository"),
        ("tests/test_contract.py::helper", "starting with 'test_'"),
        ("tests/test_contract.py::test_boundary[]", "must not be empty"),
        ("tests/test_contract.py::test_boundary[open", "parameter suffix"),
        ("tests/test_contract.py::test_boundary::extra", "exactly one '::'"),
    ],
)
def test_malformed_nodes_fail_closed(tmp_path: Path, node_id: str, message: str) -> None:
    _valid_repo(tmp_path, registry=f"NODE = {node_id!r}\n")

    with pytest.raises(StaticContractReferenceError, match=message):
        audit_static_contract_references(tmp_path)


def test_renamed_and_missing_tests_fail_closed(tmp_path: Path) -> None:
    _valid_repo(
        tmp_path,
        registry='NODE = "tests/test_contract.py::test_renamed"\n',
    )
    with pytest.raises(StaticContractReferenceError) as renamed:
        audit_static_contract_references(tmp_path)
    assert "test function not found: test_renamed" in str(renamed.value)

    _write(
        tmp_path / "src" / "registry.py",
        'NODE = "tests/test_missing.py::test_boundary"\n',
    )
    with pytest.raises(StaticContractReferenceError) as missing:
        audit_static_contract_references(tmp_path)
    assert "test file not found: tests/test_missing.py" in str(missing.value)


def test_duplicate_definitions_and_source_arguments_fail_closed(tmp_path: Path) -> None:
    _valid_repo(tmp_path)
    _write(
        tmp_path / "tests" / "test_contract.py",
        "def test_boundary():\n    pass\n\ndef test_boundary():\n    pass\n",
    )
    with pytest.raises(StaticContractReferenceError, match="duplicate top-level test"):
        audit_static_contract_references(tmp_path)

    _write(tmp_path / "tests" / "test_contract.py", "def test_boundary():\n    pass\n")
    with pytest.raises(StaticContractReferenceError, match="duplicate source argument"):
        audit_static_contract_references(tmp_path, sources=("src", "src"))


def test_symlinked_test_target_is_never_followed(tmp_path: Path) -> None:
    _valid_repo(tmp_path)
    target = tmp_path / "outside.py"
    _write(target, "def test_boundary():\n    pass\n")
    (tmp_path / "tests" / "test_contract.py").unlink()
    (tmp_path / "tests" / "test_contract.py").symlink_to(target)

    with pytest.raises(StaticContractReferenceError, match="cannot be parsed safely"):
        audit_static_contract_references(tmp_path)


@pytest.mark.parametrize(
    "forbidden",
    (
        "1234567890abcdef1234567890abcdef12345678",
        "ABCDEF1234567890ABCDEF1234567890ABCDEF12",
    ),
)
def test_current_commit_literals_are_forbidden_regardless_of_case(
    tmp_path: Path, forbidden: str
) -> None:
    _valid_repo(tmp_path)
    _write(tmp_path / "scripts" / "launch.sh", f'EXPECTED_COMMIT="{forbidden}"\n')
    with pytest.raises(StaticContractReferenceError) as captured:
        audit_static_contract_references(tmp_path)
    assert "hard-coded git commit SHA is forbidden" in str(captured.value)
    assert "--expected-commit" in str(captured.value)


def test_sha256_is_not_misclassified_as_a_git_commit(tmp_path: Path) -> None:
    _valid_repo(tmp_path)
    sha256 = "a" * 64
    uppercase_sha256 = "A" * 64
    _write(
        tmp_path / "scripts" / "launch.sh",
        f'LOWER_SHA256="{sha256}"\nUPPER_SHA256="{uppercase_sha256}"\n',
    )
    result = audit_static_contract_references(tmp_path)
    assert result.historical_sha_literals == 0


def test_only_four_historical_builder_literals_are_allowlisted(tmp_path: Path) -> None:
    _valid_repo(tmp_path)
    _write(
        tmp_path / "scripts" / "data" / "repair_source_pool_provenance.py",
        f'V1 = "{V1_BUILDER}"\nV2 = "{V2_BUILDER}"\n',
    )
    _write(
        tmp_path / "src" / "deep_anc" / "data" / "holdout_contract.py",
        f'V1 = "{V1_BUILDER}"\nV2 = "{V2_BUILDER}"\n',
    )
    result = audit_static_contract_references(tmp_path)
    assert result.historical_sha_literals == 4

    _write(tmp_path / "src" / "copied_builder.py", f'OLD = "{V1_BUILDER}"\n')
    with pytest.raises(StaticContractReferenceError, match="hard-coded git commit SHA is forbidden"):
        audit_static_contract_references(tmp_path)


def test_cli_runs_in_isolated_stdlib_mode_and_reports_stale_reference(tmp_path: Path) -> None:
    _valid_repo(tmp_path, registry='NODE = "tests/test_contract.py::test_renamed"\n')

    process = subprocess.run(
        [sys.executable, "-I", "-B", str(CHECKER), "--repo-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert process.returncode == 1
    assert FAILURE_PREFIX in process.stderr
    assert "test function not found: test_renamed" in process.stderr
    assert process.stdout == ""


def test_repository_pytest_sessionstart_calls_the_shared_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    class EarlyStop(RuntimeError):
        pass

    class Session:
        Interrupted = EarlyStop

    def fail(root: Path) -> None:
        calls.append(root)
        raise StaticContractReferenceError(("synthetic stale reference",))

    monkeypatch.setattr(repository_conftest, "audit_static_contract_references", fail)
    with pytest.raises(EarlyStop, match="synthetic stale reference"):
        repository_conftest.pytest_sessionstart(Session())
    assert calls == [REPO]


def test_pytest_hook_stops_before_collection_imports_test_modules(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "scripts" / "ci" / CHECKER.name, CHECKER.read_text(encoding="utf-8"))
    _write(tmp_path / "tests" / "conftest.py", CONFTEST.read_text(encoding="utf-8"))
    _write(
        tmp_path / "src" / "registry.py",
        'NODE = "tests/test_collection_sentinel.py::test_renamed"\n',
    )
    sentinel = tmp_path / "collection_was_reached"
    _write(
        tmp_path / "tests" / "test_collection_sentinel.py",
        (
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('bad', encoding='utf-8')\n"
            "def test_current_name():\n    pass\n"
        ),
    )

    process = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    combined = process.stdout + process.stderr
    assert process.returncode != 0
    assert FAILURE_PREFIX in combined
    assert "test function not found: test_renamed" in combined
    assert not sentinel.exists(), "stale contract must fail before test module collection/import"


def test_cli_pass_output_is_machine_visible(tmp_path: Path) -> None:
    _valid_repo(tmp_path)
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(CHECKER), "--repo-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.startswith(PASS_PREFIX)
    assert process.stderr == ""
