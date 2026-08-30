#!/usr/bin/env python3
"""운영 코드에 적힌 pytest node 참조가 실제 테스트를 가리키는지 정적으로 검사한다.

이 검사는 운영 모듈을 import 하지 않는다. 따라서 pydantic, torch 같은 프로젝트
의존성이 아직 설치되지 않은 Elice bootstrap 초기에 실행할 수 있다. 기본 검사 범위는
``src``와 ``configs``이며, ``--source``를 반복해 다른 registry/config 경로로 교체할 수
있다.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ("src", "configs")
DEFAULT_GIT_SHA_ROOTS: tuple[str, ...] = ("src", "scripts", "configs")
FAILURE_PREFIX = "[FAIL] static pytest node reference audit"
PASS_PREFIX = "[PASS] static pytest node references:"

# 현재 실행 commit은 CLI의 --expected-commit처럼 신뢰 경계 밖에서 주입해야 한다. 아래
# 두 값만 데이터 계보를 재현하는 역사 builder identity라서, 명시한 두 파일에서 각 1회만
# 허용한다. 문자열을 나눈 이유는 checker 자체가 금지 토큰을 보유하는 예외가 되지 않게
# 하기 위해서다.
_HISTORICAL_V1 = "7c7800fa94a8c5e156e0" + "49be896fd0b9586d983f"
_HISTORICAL_V2 = "0cb13b14e36c33478395" + "3aedd47aa0bc13d0fb6a"
_HISTORICAL_GIT_SHA_ALLOWLIST: dict[tuple[PurePosixPath, str], int] = {
    (PurePosixPath("scripts/data/repair_source_pool_provenance.py"), _HISTORICAL_V1): 1,
    (PurePosixPath("scripts/data/repair_source_pool_provenance.py"), _HISTORICAL_V2): 1,
    (PurePosixPath("src/deep_anc/data/holdout_contract.py"), _HISTORICAL_V1): 1,
    (PurePosixPath("src/deep_anc/data/holdout_contract.py"), _HISTORICAL_V2): 1,
}
_GIT_SHA = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")
_TEXT_CODE_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)


@dataclass(frozen=True)
class StaticReferenceAuditResult:
    """성공한 정적 감사의 집계."""

    source_files: int
    references: int
    test_files: int
    sha_scan_files: int
    historical_sha_literals: int


class StaticContractReferenceError(RuntimeError):
    """참조가 하나라도 모호하거나 깨졌을 때의 fail-closed 오류."""

    def __init__(self, issues: Iterable[str]) -> None:
        unique = tuple(dict.fromkeys(str(issue) for issue in issues))
        if not unique:
            raise ValueError("StaticContractReferenceError requires at least one issue")
        self.issues = unique
        super().__init__(self._render())

    def _render(self) -> str:
        return "\n".join((FAILURE_PREFIX, *(f"- {issue}" for issue in self.issues)))


@dataclass(frozen=True)
class _Reference:
    source: PurePosixPath
    line: int
    column: int
    node_id: str

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line}:{self.column + 1}"


def _relative_path(value: str | os.PathLike[str], *, label: str) -> PurePosixPath:
    text = os.fspath(value)
    if not text or "\\" in text:
        raise ValueError(f"{label} must be a non-empty repository-relative POSIX path: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must stay inside the repository: {text!r}")
    return path


def _secure_read_text(repo_root: Path, relative: PurePosixPath) -> str:
    """심볼릭 링크를 따라가지 않고 저장소 상대 파일을 UTF-8로 읽는다."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise OSError("O_NOFOLLOW/O_DIRECTORY is unavailable; refusing an unsafe audit read")

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    opened: list[int] = []
    try:
        current_fd = os.open(repo_root, directory_flags)
        opened.append(current_fd)
        for part in relative.parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            opened.append(current_fd)
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=current_fd)
        opened.append(file_fd)
        with os.fdopen(file_fd, "r", encoding="utf-8", errors="strict") as handle:
            opened.pop()
            return handle.read()
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _walk_python_files(
    repo_root: Path,
    source_specs: Sequence[str | os.PathLike[str]],
) -> tuple[list[PurePosixPath], list[str]]:
    issues: list[str] = []
    normalized: list[PurePosixPath] = []
    seen_specs: set[PurePosixPath] = set()
    for raw_spec in source_specs:
        try:
            spec = _relative_path(raw_spec, label="source")
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if spec in seen_specs:
            issues.append(f"duplicate source argument: {spec}")
            continue
        seen_specs.add(spec)
        normalized.append(spec)

    files: list[PurePosixPath] = []
    selected_by: dict[PurePosixPath, PurePosixPath] = {}
    for spec in normalized:
        absolute = repo_root.joinpath(*spec.parts)
        try:
            info = absolute.lstat()
        except FileNotFoundError:
            issues.append(f"source does not exist: {spec}")
            continue
        if stat.S_ISLNK(info.st_mode):
            issues.append(f"source must not be a symlink: {spec}")
            continue
        candidates: list[Path] = []
        if stat.S_ISREG(info.st_mode):
            if absolute.suffix != ".py":
                issues.append(f"source file is not Python: {spec}")
                continue
            candidates.append(absolute)
        elif stat.S_ISDIR(info.st_mode):
            for walk_root, directory_names, file_names in os.walk(absolute, followlinks=False):
                root_path = Path(walk_root)
                retained: list[str] = []
                for name in sorted(directory_names):
                    child = root_path / name
                    if child.is_symlink():
                        relative = PurePosixPath(child.relative_to(repo_root).as_posix())
                        issues.append(f"source tree contains symlink directory: {relative}")
                    else:
                        retained.append(name)
                directory_names[:] = retained
                for name in sorted(file_names):
                    if not name.endswith(".py"):
                        continue
                    child = root_path / name
                    relative = PurePosixPath(child.relative_to(repo_root).as_posix())
                    if child.is_symlink():
                        issues.append(f"source tree contains symlink file: {relative}")
                    else:
                        candidates.append(child)
        else:
            issues.append(f"source is neither a regular file nor directory: {spec}")
            continue

        for candidate in sorted(candidates):
            relative = PurePosixPath(candidate.relative_to(repo_root).as_posix())
            previous = selected_by.get(relative)
            if previous is not None:
                issues.append(
                    f"duplicate source file selected by {previous} and {spec}: {relative}"
                )
                continue
            selected_by[relative] = spec
            files.append(relative)

    if not files:
        issues.append("source scan selected no Python files")
    return sorted(files, key=str), issues


def _walk_sha_scan_files(repo_root: Path) -> tuple[list[PurePosixPath], list[str]]:
    """commit literal 정책을 적용할 운영 text 파일을 고른다."""

    files: list[PurePosixPath] = []
    issues: list[str] = []
    for root_text in DEFAULT_GIT_SHA_ROOTS:
        root = repo_root / root_text
        if not root.exists():
            continue
        try:
            root_info = root.lstat()
        except OSError as exc:
            issues.append(f"SHA scan root cannot be inspected: {root_text}: {exc}")
            continue
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            issues.append(f"SHA scan root must be a real directory: {root_text}")
            continue
        for walk_root, directory_names, file_names in os.walk(root, followlinks=False):
            root_path = Path(walk_root)
            retained: list[str] = []
            for name in sorted(directory_names):
                child = root_path / name
                relative = PurePosixPath(child.relative_to(repo_root).as_posix())
                if child.is_symlink():
                    issues.append(f"SHA scan tree contains symlink directory: {relative}")
                elif name == "__pycache__":
                    continue
                else:
                    retained.append(name)
            directory_names[:] = retained
            for name in sorted(file_names):
                child = root_path / name
                if child.suffix not in _TEXT_CODE_SUFFIXES and name != "Dockerfile":
                    continue
                relative = PurePosixPath(child.relative_to(repo_root).as_posix())
                if child.is_symlink():
                    issues.append(f"SHA scan tree contains symlink file: {relative}")
                else:
                    files.append(relative)
    return sorted(set(files), key=str), issues


def _audit_hardcoded_git_shas(
    repo_root: Path,
) -> tuple[int, int, list[str]]:
    files, issues = _walk_sha_scan_files(repo_root)
    allowed_seen: dict[tuple[PurePosixPath, str], int] = {}
    allowed_literals = 0
    for relative in files:
        try:
            text = _secure_read_text(repo_root, relative)
        except (OSError, UnicodeError) as exc:
            issues.append(f"{relative}: SHA scan file cannot be read safely: {exc}")
            continue
        for match in _GIT_SHA.finditer(text):
            value = match.group(0)
            key = (relative, value)
            allowed_limit = _HISTORICAL_GIT_SHA_ALLOWLIST.get(key, 0)
            seen = allowed_seen.get(key, 0) + 1
            allowed_seen[key] = seen
            if seen <= allowed_limit:
                allowed_literals += 1
                continue
            # bisect는 필요 없는 작은 파일이며, stdlib-only 초기 guard의 단순성을 우선한다.
            line = text.count("\n", 0, match.start()) + 1
            last_newline = text.rfind("\n", 0, match.start())
            column = match.start() - last_newline
            if allowed_limit:
                reason = "historical builder SHA exceeds its one-occurrence allowlist"
            else:
                reason = (
                    "hard-coded git commit SHA is forbidden; inject the current commit through "
                    "--expected-commit"
                )
            issues.append(f"{relative}:{line}:{column}: {value}: {reason}")
    return len(files), allowed_literals, issues


def _static_string(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, constants)
        right = _static_string(node.right, constants)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue) and value.conversion == -1:
                if value.format_spec is not None:
                    return None
                resolved = _static_string(value.value, constants)
                if resolved is None:
                    return None
                parts.append(resolved)
                continue
            return None
        return "".join(parts)
    return None


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    pending: list[tuple[str, ast.AST]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                pending.append((target.id, statement.value))
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                pending.append((statement.target.id, statement.value))

    constants: dict[str, str] = {}
    while pending:
        remaining: list[tuple[str, ast.AST]] = []
        progressed = False
        for name, expression in pending:
            value = _static_string(expression, constants)
            if value is None:
                remaining.append((name, expression))
            else:
                constants[name] = value
                progressed = True
        if not progressed:
            break
        pending = remaining
    return constants


def _docstring_nodes(tree: ast.Module) -> set[int]:
    result: set[int] = set()
    containers = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, containers) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            result.add(id(first.value))
    return result


def _looks_like_test_node(value: str) -> bool:
    stripped = value.strip()
    if not (stripped.startswith("tests/") or stripped.startswith("tests\\")):
        return False
    if "::" in stripped:
        return True
    marker = stripped.find(".py")
    return marker >= 0 and "test_" in stripped[marker + 3 :]


def _extract_references(
    source: PurePosixPath,
    tree: ast.Module,
) -> list[_Reference]:
    constants = _module_string_constants(tree)
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    docstrings = _docstring_nodes(tree)

    references: list[_Reference] = []
    expression_types = (ast.Constant, ast.JoinedStr, ast.BinOp)
    for node in ast.walk(tree):
        if not isinstance(node, expression_types) or id(node) in docstrings:
            continue
        parent = parents.get(id(node))
        if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
            continue
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add):
            continue
        value = _static_string(node, constants)
        if value is None or not _looks_like_test_node(value):
            continue
        references.append(
            _Reference(
                source=source,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0),
                node_id=value,
            )
        )
    return sorted(references, key=lambda item: (item.line, item.column, item.node_id))


def _parse_node_id(node_id: str) -> tuple[PurePosixPath, str]:
    if node_id != node_id.strip() or any(character in node_id for character in "\r\n\0\\"):
        raise ValueError("malformed node id: whitespace, NUL, or backslash is forbidden")
    if node_id.count("::") != 1:
        raise ValueError("malformed node id: exactly one '::' separator is required")
    file_text, test_text = node_id.split("::", 1)
    try:
        test_file = _relative_path(file_text, label="test file")
    except ValueError as exc:
        raise ValueError(f"malformed node id: {exc}") from exc
    if test_file.parts[0] != "tests" or test_file.suffix != ".py":
        raise ValueError("malformed node id: test file must be tests/**/*.py")
    if not test_file.name.startswith("test_"):
        raise ValueError("malformed node id: test module basename must start with 'test_'")

    if "[" in test_text or "]" in test_text:
        open_count = test_text.count("[")
        close_count = test_text.count("]")
        if open_count != 1 or close_count != 1 or not test_text.endswith("]"):
            raise ValueError("malformed node id: parameter suffix must be one non-empty '[...]'")
        test_name, parameter = test_text[:-1].split("[", 1)
        if not parameter:
            raise ValueError("malformed node id: parameter suffix must not be empty")
    else:
        test_name = test_text
    if not test_name.startswith("test_") or not test_name.isidentifier():
        raise ValueError("malformed node id: function must be an identifier starting with 'test_'")
    return test_file, test_name


def _top_level_tests(tree: ast.Module) -> tuple[set[str], set[str]]:
    counts: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith(
            "test_"
        ):
            counts[statement.name] = counts.get(statement.name, 0) + 1
    return set(counts), {name for name, count in counts.items() if count > 1}


def audit_static_contract_references(
    repo_root: str | os.PathLike[str],
    sources: Sequence[str | os.PathLike[str]] | None = None,
) -> StaticReferenceAuditResult:
    """운영 소스의 정적 pytest node 참조를 검사한다.

    오류가 하나라도 있으면 :class:`StaticContractReferenceError`를 던진다. 동일 node를
    여러 gate가 공유하는 것은 허용하지만, 각 literal은 독립적으로 resolve한다.
    """

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise StaticContractReferenceError((f"repository root is not a directory: {root}",))
    source_specs = tuple(DEFAULT_SOURCE_ROOTS if sources is None else sources)
    source_files, issues = _walk_python_files(root, source_specs)
    sha_scan_files, historical_sha_literals, sha_issues = _audit_hardcoded_git_shas(root)
    issues.extend(sha_issues)

    references: list[_Reference] = []
    for source in source_files:
        try:
            text = _secure_read_text(root, source)
            tree = ast.parse(text, filename=str(source))
        except (OSError, UnicodeError, SyntaxError) as exc:
            issues.append(f"{source}: source cannot be parsed safely: {exc}")
            continue
        references.extend(_extract_references(source, tree))

    if not references:
        issues.append("source scan found no static pytest node references")

    test_cache: dict[PurePosixPath, tuple[set[str], set[str]] | str] = {}
    referenced_test_files: set[PurePosixPath] = set()
    for reference in references:
        try:
            test_file, test_name = _parse_node_id(reference.node_id)
        except ValueError as exc:
            issues.append(f"{reference.location}: {reference.node_id!r}: {exc}")
            continue
        referenced_test_files.add(test_file)

        cached = test_cache.get(test_file)
        if cached is None:
            absolute = root.joinpath(*test_file.parts)
            if not absolute.exists():
                cached = f"test file not found: {test_file}"
            elif not absolute.is_file():
                cached = f"test target is not a regular file: {test_file}"
            else:
                try:
                    target_text = _secure_read_text(root, test_file)
                    target_tree = ast.parse(target_text, filename=str(test_file))
                    cached = _top_level_tests(target_tree)
                except (OSError, UnicodeError, SyntaxError) as exc:
                    cached = f"test file cannot be parsed safely: {test_file}: {exc}"
            test_cache[test_file] = cached

        if isinstance(cached, str):
            issues.append(f"{reference.location}: {reference.node_id!r}: {cached}")
            continue
        functions, duplicates = cached
        if duplicates:
            names = ", ".join(sorted(duplicates))
            issues.append(
                f"{reference.location}: {reference.node_id!r}: duplicate top-level test "
                f"function definition(s) in {test_file}: {names}"
            )
            continue
        if test_name not in functions:
            issues.append(
                f"{reference.location}: {reference.node_id!r}: "
                f"test function not found: {test_name}"
            )

    if issues:
        raise StaticContractReferenceError(issues)
    return StaticReferenceAuditResult(
        source_files=len(source_files),
        references=len(references),
        test_files=len(referenced_test_files),
        sha_scan_files=sha_scan_files,
        historical_sha_literals=historical_sha_literals,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: checker script repository)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="repository-relative Python file/directory; repeat to replace src/configs defaults",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = audit_static_contract_references(arguments.repo_root, arguments.source)
    except (OSError, StaticContractReferenceError) as exc:
        if isinstance(exc, StaticContractReferenceError):
            message = str(exc)
        else:
            message = f"{FAILURE_PREFIX}\n- repository cannot be audited safely: {exc}"
        print(message, file=sys.stderr)
        return 1
    print(
        f"{PASS_PREFIX} sources={result.source_files}, "
        f"references={result.references}, test_files={result.test_files}, "
        f"sha_scan_files={result.sha_scan_files}, "
        f"historical_sha_literals={result.historical_sha_literals}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
