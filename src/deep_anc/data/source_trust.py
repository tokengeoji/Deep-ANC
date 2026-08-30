"""공식 artifact 발행 시 clean exact Git source를 독립 검증한다."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable


SOURCE_TRUST_SCHEMA = "exact_clean_git_source/v1"
SELECTOR_RUNTIME_SCHEMA = "isolated_dns_selector_runtime/v2"
PROTECTED_IGNORED_ROOTS = ("src", "scripts", "configs")
SELECTOR_PYCACHE_PREFIX = "/dev/null/deep-anc-selector"
GIT_EXECUTABLE = "/usr/bin/git"
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PROJECT_REQUIREMENT_RE = re.compile(
    r"^deep[-_.]anc(?:\s*@\s*\S+|==\S+)$",
    re.IGNORECASE,
)


class SourceTrustError(ValueError):
    """현재 source가 clean exact commit으로 증명되지 않는다."""


def validate_environment_freeze_source_commit(
    raw: bytes,
    *,
    expected_commit: str,
) -> str:
    """``pip freeze``의 유일한 editable Deep-ANC가 exact commit인지 검증한다.

    Editable 설치는 현재 checkout의 Python source를 즉시 따라가지만, 과거에 저장한
    ``pip freeze`` 파일은 이전 checkout SHA를 계속 담을 수 있다. 따라서 freeze의
    파일 SHA만 결속해서는 code/environment 조합을 재현했다고 볼 수 없다.

    이 검증기는 축약 SHA, missing/duplicate project requirement, non-editable local
    package와 stale VCS revision을 모두 거부한다. 다른 package의 editable requirement는
    Deep-ANC identity로 오인하지 않는다. 반환값은 검증된 requirement 한 줄이다.
    """

    expected = str(expected_commit)
    if _FULL_COMMIT_RE.fullmatch(expected) is None:
        raise SourceTrustError(
            "environment freeze expected commit은 소문자 전체 40자리 SHA여야 합니다"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceTrustError("environment freeze receipt가 UTF-8이 아닙니다") from exc

    project_lines: list[str] = []
    for original in lines:
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if "#egg=deep_anc" in lowered or "#egg=deep-anc" in lowered:
            project_lines.append(line)
            continue
        if _PROJECT_REQUIREMENT_RE.fullmatch(line) is not None:
            project_lines.append(line)

    if len(project_lines) != 1:
        raise SourceTrustError(
            "environment freeze에는 Deep-ANC project requirement가 정확히 하나여야 "
            f"합니다: count={len(project_lines)}"
        )
    requirement = project_lines[0]
    if not requirement.startswith("-e git+"):
        raise SourceTrustError(
            "environment freeze의 Deep-ANC는 editable VCS requirement여야 합니다"
        )
    url, separator, fragment = requirement[3:].partition("#")
    if not separator:
        raise SourceTrustError("Deep-ANC editable VCS requirement에 egg fragment가 없습니다")
    egg_values = []
    for item in fragment.split("&"):
        key, equals, value = item.partition("=")
        if equals and key.lower() == "egg":
            egg_values.append(value.lower().replace("_", "-").replace(".", "-"))
    if egg_values != ["deep-anc"]:
        raise SourceTrustError(
            "Deep-ANC editable VCS requirement의 egg identity가 exact하지 않습니다"
        )
    _remote, revision_separator, revision = url.rpartition("@")
    if not revision_separator or _FULL_COMMIT_RE.fullmatch(revision) is None:
        raise SourceTrustError(
            "Deep-ANC editable VCS requirement는 전체 40자리 revision을 가져야 합니다"
        )
    if revision != expected:
        raise SourceTrustError(
            "environment freeze Deep-ANC commit이 expected checkout과 다릅니다: "
            f"freeze={revision}, expected={expected}"
        )
    return requirement


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, arguments: Iterable[str], *, check: bool = True) -> bytes:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    git_dir = root / ".git"
    try:
        result = subprocess.run(
            [
                GIT_EXECUTABLE,
                f"--git-dir={git_dir}",
                f"--work-tree={root}",
                "-c",
                f"core.worktree={root}",
                *arguments,
            ],
            cwd=root,
            env=environment,
            check=check,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceTrustError(
            f"clean source Git 검증 명령 실패: git {' '.join(arguments)}: {exc}"
        ) from exc
    return bytes(result.stdout)


def _untracked_protected_paths(
    root: Path,
    tracked: set[str],
    *,
    reject_runtime_bytecode: bool,
) -> list[str]:
    """Git ignore/ambient worktree와 독립적으로 import 가능 tree를 직접 열거한다."""

    unexpected: list[str] = []
    for relative_root in PROTECTED_IGNORED_ROOTS:
        protected = root / relative_root
        if not protected.exists():
            continue
        for directory, names, filenames in os.walk(protected, followlinks=False):
            base = Path(directory)
            # editable install이 만드는 ``*.egg-info``는 import 실행 경로가 아니며
            # bootstrap 뒤에도 정상 존재한다. 반면 __pycache__/.pyc는 실행 가능하므로
            # 면제하지 않는다. egg-info symlink 자체도 아래 direct check가 거부한다.
            names[:] = [
                name
                for name in names
                if not (
                    name.endswith(".egg-info")
                    and (base / name).is_dir()
                    and not (base / name).is_symlink()
                )
            ]
            for name in [*names, *filenames]:
                path = base / name
                if path.is_dir() and not path.is_symlink():
                    continue
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError as exc:
                    raise SourceTrustError(
                        f"protected source path가 repository 밖입니다: {path}"
                    ) from exc
                if relative in tracked:
                    continue
                is_runtime_cache = (
                    not path.is_symlink()
                    and path.is_file()
                    and path.suffix == ".pyc"
                    and "__pycache__" in path.parts
                )
                if is_runtime_cache and not reject_runtime_bytecode:
                    continue
                if relative not in tracked:
                    unexpected.append(relative)
    return sorted(set(unexpected))


def _allowed_ignored_protected_path(
    relative: str, *, reject_runtime_bytecode: bool
) -> bool:
    path = Path(relative)
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    return bool(
        not reject_runtime_bytecode
        and path.suffix == ".pyc"
        and "__pycache__" in path.parts
    )


def canonical_selector_sys_path(repo_root: str | Path) -> tuple[str, ...]:
    """``python -I -S -B`` selector가 사용할 유일한 import search path."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    zip_path = stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    candidates = [
        root / "src",
        zip_path,
        stdlib,
        stdlib / "lib-dynload",
        root / ".venv/lib" / version / "site-packages",
        Path(sys.base_prefix) / "local/lib" / version / "dist-packages",
        Path(sys.base_prefix) / "lib" / version / "dist-packages",
        Path(sys.base_prefix) / "lib/python3/dist-packages",
    ]
    values: list[str] = []
    for index, candidate in enumerate(candidates):
        # CPython은 존재하지 않는 stdlib zip도 canonical sys.path에 둔다.
        if index == 1 or candidate.is_dir():
            value = os.path.abspath(os.fspath(candidate))
            if value not in values:
                values.append(value)
    return tuple(values)


def _runtime_module_ref(
    name: str, *, allowed_roots: tuple[Path, ...]
) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise SourceTrustError(f"selector runtime module import 실패: {name}: {exc}") from exc
    raw_path = getattr(module, "__file__", None)
    specification = getattr(module, "__spec__", None)
    origin = getattr(specification, "origin", None)
    loader = getattr(module, "__loader__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise SourceTrustError(f"selector runtime module file이 없습니다: {name}")
    if not isinstance(origin, str) or not origin:
        raise SourceTrustError(f"selector runtime module origin이 없습니다: {name}")
    try:
        path = Path(raw_path).resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise SourceTrustError(f"selector runtime module file 검증 실패: {name}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SourceTrustError(f"selector runtime module은 regular file이어야 합니다: {name}")
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise SourceTrustError(
            f"selector runtime module이 canonical sys.path 밖입니다: {name}={path}"
        )
    try:
        origin_path = Path(origin).resolve(strict=True)
    except OSError as exc:
        raise SourceTrustError(
            f"selector runtime module origin 검증 실패: {name}: {exc}"
        ) from exc
    if origin_path != path:
        raise SourceTrustError(
            f"selector runtime module __file__/loader origin이 다릅니다: {name}"
        )

    source_loader = isinstance(loader, importlib.machinery.SourceFileLoader)
    extension_loader = isinstance(loader, importlib.machinery.ExtensionFileLoader)
    if source_loader:
        origin_kind = "source"
    elif extension_loader:
        origin_kind = "native_extension"
    else:
        raise SourceTrustError(
            f"selector runtime module loader를 신뢰할 수 없습니다: "
            f"{name}={type(loader).__name__}"
        )

    cached = getattr(module, "__cached__", None)
    cached_path: str | None = None
    if cached is not None:
        if not isinstance(cached, str) or not cached:
            raise SourceTrustError(f"selector runtime module __cached__가 유효하지 않습니다: {name}")
        cached_path = os.path.abspath(cached)
        prefix = os.path.abspath(SELECTOR_PYCACHE_PREFIX) + os.sep
        if not cached_path.startswith(prefix):
            raise SourceTrustError(
                f"selector runtime module이 adjacent/ambient bytecode를 참조합니다: "
                f"{name}={cached_path}"
            )
        # canonical prefix는 /dev/null 아래라 실제 bytecode가 존재할 수 없다.
        if os.path.lexists(cached_path):
            raise SourceTrustError(
                f"selector canonical bytecode prefix에 실제 cache가 존재합니다: {name}"
            )
    if source_loader and cached_path is None:
        raise SourceTrustError(
            f"selector source module의 canonical __cached__ evidence가 없습니다: {name}"
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceTrustError(f"selector runtime module hash 실패: {name}: {exc}") from exc
    version = getattr(module, "__version__", None)
    return {
        "name": name,
        "path": os.path.abspath(os.fspath(path)),
        "sha256": digest.hexdigest(),
        "size": int(info.st_size),
        "version": str(version) if version is not None else None,
        "loader": type(loader).__name__,
        "origin_kind": origin_kind,
        "cached_path": cached_path,
    }


def _runtime_libsndfile_ref(
    soundfile_module: Any, *, allowed_roots: tuple[Path, ...]
) -> dict[str, Any]:
    """SoundFile이 실제 dlopen한 packaged libsndfile bytes를 봉인한다."""

    raw_directory = getattr(soundfile_module, "_path", None)
    version = getattr(soundfile_module, "__libsndfile_version__", None)
    loaded = getattr(soundfile_module, "_snd", None)
    if not isinstance(raw_directory, str) or not raw_directory or not isinstance(version, str):
        raise SourceTrustError("SoundFile packaged libsndfile path/version evidence가 없습니다")
    try:
        directory = Path(raw_directory).resolve(strict=True)
    except OSError as exc:
        raise SourceTrustError(f"SoundFile backend directory 검증 실패: {exc}") from exc
    if not directory.is_dir() or directory.is_symlink() or not any(
        directory == root or root in directory.parents for root in allowed_roots
    ):
        raise SourceTrustError("SoundFile backend directory가 canonical runtime root 밖입니다")
    candidates = sorted(
        path
        for pattern in ("libsndfile_*.so", "libsndfile_*.dylib", "libsndfile_*.dll")
        for path in directory.glob(pattern)
    )
    if len(candidates) != 1:
        raise SourceTrustError(
            f"SoundFile packaged libsndfile backend은 정확히 1개여야 합니다: {candidates}"
        )
    path = candidates[0].resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SourceTrustError("SoundFile packaged libsndfile backend이 regular file이 아닙니다")
    # cffi Lib object는 public path attribute를 제공하지 않으므로, SoundFile이
    # 생성한 object representation에서 실제 dlopen target을 추가로 대조한다.
    if loaded is None or os.path.abspath(path) not in repr(loaded):
        raise SourceTrustError("SoundFile cffi object가 선언된 packaged libsndfile을 로드하지 않았습니다")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": os.path.abspath(path),
        "sha256": digest.hexdigest(),
        "size": int(info.st_size),
        "version": version,
    }


def _freeze_versions(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceTrustError("environment freeze receipt가 UTF-8이 아닙니다") from exc
    versions: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        canonical = name.strip().lower().replace("_", "-")
        if canonical in versions:
            raise SourceTrustError(f"environment freeze package가 중복됩니다: {canonical}")
        versions[canonical] = version.strip()
    return versions


def exact_selector_runtime_evidence(
    repo_root: str | Path,
    *,
    freeze_receipt: str | Path,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """DNS full-scan interpreter/import 환경을 live bytes에서 fail-closed 봉인한다."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    flags = {
        "isolated": int(sys.flags.isolated),
        "ignore_environment": int(sys.flags.ignore_environment),
        "no_user_site": int(sys.flags.no_user_site),
        "no_site": int(sys.flags.no_site),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
    }
    if flags != {
        "isolated": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "no_site": 1,
        "dont_write_bytecode": 1,
    }:
        raise SourceTrustError(
            "DNS selector는 canonical .venv/bin/python -I -S -B "
            "-X pycache_prefix=/dev/null/deep-anc-selector로만 실행해야 합니다"
        )
    expected_executable = os.path.abspath(os.fspath(root / ".venv/bin/python"))
    if os.path.abspath(sys.executable) != expected_executable:
        raise SourceTrustError(
            "DNS selector interpreter가 repository canonical .venv/bin/python이 아닙니다"
        )
    try:
        executable_realpath = Path(sys.executable).resolve(strict=True)
        executable_info = executable_realpath.stat()
        executable_raw = executable_realpath.read_bytes()
    except OSError as exc:
        raise SourceTrustError(f"DNS selector interpreter bytes 검증 실패: {exc}") from exc
    if not stat.S_ISREG(executable_info.st_mode):
        raise SourceTrustError("DNS selector interpreter target은 regular file이어야 합니다")
    expected_path = canonical_selector_sys_path(root)
    actual_path = tuple(os.path.abspath(value) for value in sys.path)
    if actual_path != expected_path:
        raise SourceTrustError(
            "DNS selector sys.path가 canonical isolated search path와 다릅니다"
        )
    for forbidden in ("sitecustomize", "usercustomize"):
        if forbidden in sys.modules:
            raise SourceTrustError(
                f"DNS selector에 금지된 ambient/module 경로가 이미 import됐습니다: {forbidden}"
            )
    if sys.pycache_prefix != SELECTOR_PYCACHE_PREFIX:
        raise SourceTrustError(
            "DNS selector는 canonical -X pycache_prefix=/dev/null/"
            "deep-anc-selector로 adjacent bytecode 재사용을 차단해야 합니다"
        )

    freeze_path = Path(freeze_receipt)
    if not freeze_path.is_absolute():
        freeze_path = root / freeze_path
    try:
        freeze_path = freeze_path.resolve(strict=True)
        freeze_raw = freeze_path.read_bytes()
    except OSError as exc:
        raise SourceTrustError(f"environment freeze receipt 읽기 실패: {exc}") from exc
    freeze_sha = hashlib.sha256(freeze_raw).hexdigest()
    if freeze_sha != str(expected_freeze_sha256).lower():
        raise SourceTrustError(
            "live environment freeze receipt가 bootstrap SHA와 다릅니다"
        )

    allowed_roots = tuple(
        Path(value).resolve()
        for value in expected_path
        if Path(value).is_dir()
    )
    numpy_module = importlib.import_module("numpy")
    # 사용하는 FFT backend module을 lazy import까지 끝낸 뒤 실제 로드 집합을 봉인한다.
    numpy_module.fft.rfft(numpy_module.asarray([0.0, 1.0], dtype=numpy_module.float64))
    importlib.import_module("scipy.signal")
    fft_modules = sorted(
        name for name in sys.modules if name == "numpy.fft" or name.startswith("numpy.fft.")
    )
    numpy_native_modules = sorted(
        name
        for name, module in sys.modules.items()
        if name.startswith("numpy")
        and isinstance(getattr(module, "__file__", None), str)
        and isinstance(getattr(module, "__loader__", None), importlib.machinery.ExtensionFileLoader)
    )
    scipy_native_modules = sorted(
        name
        for name, module in sys.modules.items()
        if name.startswith("scipy")
        and isinstance(getattr(module, "__file__", None), str)
        and isinstance(
            getattr(module, "__loader__", None),
            importlib.machinery.ExtensionFileLoader,
        )
    )
    module_names = [
        "numpy",
        *fft_modules,
        *numpy_native_modules,
        "soundfile",
        "_soundfile",
        "_cffi_backend",
        "scipy",
        "scipy.signal",
        *scipy_native_modules,
    ]
    module_refs = {
        name: _runtime_module_ref(name, allowed_roots=allowed_roots)
        for name in dict.fromkeys(module_names)
    }
    soundfile_module = importlib.import_module("soundfile")
    libsndfile = _runtime_libsndfile_ref(
        soundfile_module, allowed_roots=allowed_roots
    )
    versions = _freeze_versions(freeze_raw)
    for package, module_name in (
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("scipy", "scipy"),
    ):
        live = module_refs[module_name]["version"]
        if live is None or versions.get(package) != live:
            raise SourceTrustError(
                f"live {package} version이 freeze receipt와 다릅니다: "
                f"live={live}, freeze={versions.get(package)}"
            )
    return {
        "schema": SELECTOR_RUNTIME_SCHEMA,
        "python_executable": expected_executable,
        "python_executable_realpath": os.path.abspath(executable_realpath),
        "python_executable_sha256": hashlib.sha256(executable_raw).hexdigest(),
        "python_executable_size": int(executable_info.st_size),
        "python_base_prefix": os.path.abspath(sys.base_prefix),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "flags": flags,
        "pycache_prefix": SELECTOR_PYCACHE_PREFIX,
        "sys_path": list(expected_path),
        "environment_freeze_sha256": freeze_sha,
        "modules": module_refs,
        "libsndfile": libsndfile,
        "scipy_policy": "provenance_recorded_never_called_by_dns_numpy_power2_fft",
    }


def _parse_tree(raw: bytes, *, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path_bytes = record.split(b"\t", 1)
            fields = metadata.decode("ascii").split()
            if label == "HEAD":
                mode, kind, object_id = fields
                stage = "0"
            else:
                mode, object_id, stage = fields
                kind = "blob"
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SourceTrustError(f"{label} tree inventory를 파싱할 수 없습니다") from exc
        if kind != "blob" or stage != "0" or mode not in {"100644", "100755", "120000"}:
            raise SourceTrustError(
                f"{label} tree에 지원하지 않는 object가 있습니다: "
                f"path={path}, mode={mode}, kind={kind}, stage={stage}"
            )
        rows.append({"path": path, "mode": mode, "object_id": object_id})
    rows.sort(key=lambda item: item["path"])
    if len(rows) != len({item["path"] for item in rows}):
        raise SourceTrustError(f"{label} tree path가 중복됩니다")
    return rows


def _blob_object_id(path: Path, *, mode: str, algorithm: str) -> str:
    try:
        info = path.lstat()
        if mode == "120000":
            if not stat.S_ISLNK(info.st_mode):
                raise OSError("expected symlink")
            chunks = [os.fsencode(os.readlink(path))]
            size = len(chunks[0])
        else:
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise OSError("expected regular file")
            executable = bool(info.st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise OSError(f"executable mode mismatch: expected={mode}")
            size = int(info.st_size)
            chunks = None
    except OSError as exc:
        raise SourceTrustError(f"tracked source path 검증 실패: {path}: {exc}") from exc

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise SourceTrustError(f"지원하지 않는 Git object format입니다: {algorithm}") from exc
    digest.update(f"blob {size}\0".encode("ascii"))
    if chunks is not None:
        digest.update(chunks[0])
    else:
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise SourceTrustError(f"tracked source bytes 읽기 실패: {path}: {exc}") from exc
    return digest.hexdigest()


def exact_clean_source_evidence(
    repo_root: str | Path,
    *,
    expected_commit: str | None = None,
    reject_runtime_bytecode: bool = False,
) -> dict[str, Any]:
    """HEAD/index/worktree 및 관련 untracked injection을 fail-closed 검증한다.

    ``data/``·``results/`` 같은 ignored artifact는 Git ignore 정책대로 허용하지만,
    ``src/``·``scripts/``·``configs/`` 아래 ignored untracked executable도 별도 거부한다.
    일반 validator는 import가 만든 regular ``__pycache__/*.pyc``만 허용하고, 공식
    issuer는 ``reject_runtime_bytecode=True``로 이 cache까지 import 전에 거부한다.
    tracked 파일은 status에만 의존하지 않고 HEAD/index object와 실제 blob bytes/mode를
    모두 대조한다.
    """

    try:
        root = Path(repo_root).expanduser().resolve(strict=True)
        git_info = (root / ".git").lstat()
    except OSError as exc:
        raise SourceTrustError(f"repository root/.git을 확인할 수 없습니다: {exc}") from exc
    if not stat.S_ISDIR(git_info.st_mode) or (root / ".git").is_symlink():
        raise SourceTrustError("exact source는 root/.git 실제 directory checkout이어야 합니다")
    try:
        commit = _git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode(
            "ascii"
        ).strip().lower()
        tree_id = _git(root, ["rev-parse", "--verify", "HEAD^{tree}"]).decode(
            "ascii"
        ).strip().lower()
        object_format = _git(root, ["rev-parse", "--show-object-format"]).decode(
            "ascii"
        ).strip().lower()
        top_level = _git(root, ["rev-parse", "--show-toplevel"]).decode(
            "utf-8"
        ).strip()
        absolute_git_dir = _git(root, ["rev-parse", "--absolute-git-dir"]).decode(
            "utf-8"
        ).strip()
    except UnicodeDecodeError as exc:
        raise SourceTrustError("Git commit/tree/object format이 ASCII가 아닙니다") from exc
    if expected_commit is not None and commit != str(expected_commit).lower():
        raise SourceTrustError(
            f"clean source HEAD가 expected commit과 다릅니다: {commit} != {expected_commit}"
        )
    if Path(top_level).resolve() != root or Path(absolute_git_dir).resolve() != root / ".git":
        raise SourceTrustError(
            "Git top-level/metadata directory가 requested repository root와 다릅니다"
        )
    replace_refs = _git(root, ["replace", "-l"])
    if replace_refs.strip():
        raise SourceTrustError("git replace ref가 있어 exact source를 신뢰할 수 없습니다")
    git_dir_raw = _git(root, ["rev-parse", "--absolute-git-dir"])
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise SourceTrustError("Git directory path가 UTF-8이 아닙니다") from exc
    grafts = git_dir / "info/grafts"
    try:
        if grafts.is_file() and grafts.stat().st_size > 0:
            raise SourceTrustError("legacy git grafts가 있어 exact source를 신뢰할 수 없습니다")
    except OSError as exc:
        raise SourceTrustError(f"git grafts 검증 실패: {exc}") from exc

    status = _git(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    if status:
        preview = [
            value.decode("utf-8", errors="backslashreplace")
            for value in status.split(b"\0")[:5]
            if value
        ]
        raise SourceTrustError(
            f"tracked/staged/non-ignored untracked 변경이 있습니다: {preview}"
        )
    flags = _git(root, ["ls-files", "-v", "-z"])
    suspicious_flags = [
        record
        for record in flags.split(b"\0")
        if record and (record[:1].islower() or record[:1] == b"S")
    ]
    if suspicious_flags:
        raise SourceTrustError("assume-unchanged/skip-worktree index flag가 있습니다")
    ignored_injection = _git(
        root,
        [
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *PROTECTED_IGNORED_ROOTS,
        ],
    )
    ignored_paths = [
        value.decode("utf-8", errors="backslashreplace")
        for value in ignored_injection.split(b"\0")
        if value
    ]
    forbidden_ignored = [
        value
        for value in ignored_paths
        if not _allowed_ignored_protected_path(
            value, reject_runtime_bytecode=reject_runtime_bytecode
        )
    ]
    if forbidden_ignored:
        preview = forbidden_ignored[:5]
        raise SourceTrustError(
            f"protected source root에 ignored untracked injection이 있습니다: {preview}"
        )

    head_rows = _parse_tree(
        _git(root, ["ls-tree", "-r", "-z", "--full-tree", commit]),
        label="HEAD",
    )
    index_rows = _parse_tree(
        _git(root, ["ls-files", "--stage", "-z"]),
        label="index",
    )
    if index_rows != head_rows:
        raise SourceTrustError("Git index tree가 HEAD tree와 exact 일치하지 않습니다")
    unexpected_protected = _untracked_protected_paths(
        root,
        {row["path"] for row in head_rows},
        reject_runtime_bytecode=reject_runtime_bytecode,
    )
    if unexpected_protected:
        raise SourceTrustError(
            "실제 protected source tree에 untracked injection이 있습니다: "
            f"{unexpected_protected[:5]}"
        )
    for row in head_rows:
        actual = _blob_object_id(
            root / row["path"], mode=row["mode"], algorithm=object_format
        )
        if actual != row["object_id"]:
            raise SourceTrustError(
                "tracked worktree bytes가 HEAD blob과 다릅니다: "
                f"{row['path']} ({actual} != {row['object_id']})"
            )

    return {
        "schema": SOURCE_TRUST_SCHEMA,
        "commit": commit,
        "head_tree_object_id": tree_id,
        "git_object_format": object_format,
        "tracked_file_count": len(head_rows),
        "tracked_inventory_sha256": _canonical_json_sha256(head_rows),
        "policy": {
            "tracked_worktree": "exact_HEAD_blob_and_mode",
            "index": "exact_HEAD_tree_no_hidden_flags",
            "nonignored_untracked": "forbidden",
            "protected_ignored_roots": list(PROTECTED_IGNORED_ROOTS),
            "protected_runtime_bytecode": (
                "forbidden"
                if reject_runtime_bytecode
                else "allowed_only_regular_pyc_below___pycache__"
            ),
            "ignored_artifacts_outside_protected_roots": "allowed",
            "replace_refs_and_grafts": "forbidden",
        },
    }


__all__ = [
    "PROTECTED_IGNORED_ROOTS",
    "GIT_EXECUTABLE",
    "SELECTOR_RUNTIME_SCHEMA",
    "SELECTOR_PYCACHE_PREFIX",
    "SOURCE_TRUST_SCHEMA",
    "SourceTrustError",
    "canonical_selector_sys_path",
    "exact_clean_source_evidence",
    "exact_selector_runtime_evidence",
]
