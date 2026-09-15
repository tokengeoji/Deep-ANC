"""ESC-50 패키징의 모의 Git/임시 fixture 회귀. 실제 Git/네트워크/원본 접근 없음."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from deep_anc.data.archive_staging import validate_staging_provenance


SECRET = "TEST_ONLY_SECRET_NOT_REAL"
TREE = "a" * 40


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/data/package_esc50_git.py"
    spec = importlib.util.spec_from_file_location("esc50_git_package_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    return module


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "fixture_repo"
    (path / ".git/objects/info").mkdir(parents=True)
    (path / ".git/objects/pack").mkdir()
    (path / ".git/config").write_text("[core]\n\tbare = false\n")
    (path / ".git/HEAD").write_text("ref: refs/heads/master\n")
    (path / "untracked_secret_fixture").write_text(SECRET)
    (path / "README.md").write_text("fixture only\n")
    return path


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


class BoundedStream(io.BytesIO):
    def __init__(self, value, *, fail=False):
        super().__init__(value)
        self.requests = []
        self.fail = fail

    def read(self, size=-1):
        assert 0 < size <= 1024 * 1024
        self.requests.append(size)
        if self.fail and len(self.requests) > 1:
            raise OSError("fixture interrupted stream")
        # 짧은 read도 정상 스트림의 일부다. 한 번의 read로 전체를 요구하지 않는다.
        return super().read(min(size, 4093))


class FakeProcess:
    def __init__(self, data, *, returncode=0, fail_read=False):
        self.stdout = BoundedStream(data, fail=fail_read)
        self.returncode = returncode
        self.killed = False
        self.waits = 0

    def wait(self):
        self.waits += 1
        return self.returncode

    def kill(self):
        self.killed = True


class GitFixture:
    def __init__(self, module):
        self.module = module
        self.calls = []
        self.archive_calls = []
        self.remote = module.SOURCE
        self.head = module.COMMIT
        self.tree = TREE
        self.local_config = "core.bare\nfalse\0remote.origin.url\n" + module.SOURCE + "\0"
        self.fail_action = None
        self.returncode = 0
        self.fail_read = False
        self.process = None
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            value = b"fixture tracked content\n"
            info = tarfile.TarInfo("ESC-50/README.md")
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
        self.tar = buffer.getvalue()

    @staticmethod
    def action(command):
        return tuple(command[command.index("-C") + 2:])

    def run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        action = self.action(command)
        if action == self.fail_action:
            raise subprocess.CalledProcessError(1, command, output=SECRET, stderr=SECRET)
        answers = {
            ("config", "--local", "--no-includes", "--null", "--list"): self.local_config,
            ("remote", "get-url", "origin"): self.remote,
            ("rev-parse", "HEAD"): self.head,
            ("fsck", "--full", "--strict"): "",
            ("rev-parse", self.module.COMMIT + "^{tree}"): self.tree,
        }
        assert action in answers, f"허용되지 않은 모의 Git 호출: {action}"
        return SimpleNamespace(stdout=answers[action])

    def popen(self, command, **kwargs):
        self.archive_calls.append((command, kwargs))
        assert self.action(command) == ("archive", "--format=tar", "--prefix=ESC-50/", self.module.COMMIT)
        self.process = FakeProcess(self.tar, returncode=self.returncode, fail_read=self.fail_read)
        return self.process


@pytest.fixture
def git(cli, monkeypatch):
    fake = GitFixture(cli)
    monkeypatch.setattr(cli.subprocess, "run", fake.run)
    monkeypatch.setattr(cli.subprocess, "Popen", fake.popen)
    return fake


def test_success_preserves_repo_and_emits_git_not_publisher_checksum(cli, repo, git, tmp_path):
    before = snapshot(repo)
    out = tmp_path / "new_archive"
    receipt = cli.package(repo, out)
    archive = Path(receipt["archive_path"])
    data = archive.read_bytes()
    assert gzip.decompress(data) == git.tar
    assert receipt["name"] == f"ESC-50-{cli.COMMIT}.tar.gz"
    assert receipt["bytes"] == receipt["size"] == len(data)
    assert receipt["sha256"] == receipt["source_checksum"] == hashlib.sha256(data).hexdigest()
    assert receipt["md5"] == hashlib.md5(data).hexdigest()
    assert receipt["git_commit"] == cli.COMMIT and receipt["git_tree"] == TREE
    assert receipt["git_source_url"] == cli.SOURCE and receipt["git_fsck_verified"]
    assert receipt["source_content_verified"] and receipt["source_checksum_verified"] is False
    assert receipt["checksum_origin"] == "locally_generated_git_archive_not_publisher_archive_digest"
    provenance = validate_staging_provenance(receipt)
    assert provenance["source_checksum_scope"] == "locally_generated_archive"
    assert provenance["provenance_attestation_only"]
    assert "CC BY-NC 3.0" in receipt["license"]
    assert receipt["research_use"] == "noncommercial_academic"
    assert not any(receipt[key] for key in ("drive_upload_verified", "local_deletion_performed",
                                          "pcm_qa_verified", "extraction_performed"))
    assert receipt["temporary_local_staging"] and receipt["delete_after_verified_upload_required"]
    assert json.loads((out / "receipt.json").read_text()) == receipt
    assert SECRET not in json.dumps(receipt) and SECRET.encode() not in gzip.decompress(data)
    assert snapshot(repo) == before
    assert git.process.stdout.closed and not git.process.killed
    assert git.process.stdout.requests and max(git.process.stdout.requests) <= 1024 * 1024


def test_commands_are_read_only_and_environment_does_not_inherit_secrets(cli, repo, git, tmp_path, monkeypatch):
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0",
                "GIT_CONFIG_VALUE_0", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_SSH_COMMAND", "API_TOKEN"):
        monkeypatch.setenv(key, SECRET)
    monkeypatch.setenv("PATH", "/fixture/bin")
    monkeypatch.setenv("LANG", "C")
    cli.package(repo, tmp_path / "new")
    expected_environment = {
        "PATH": "/fixture/bin", "LANG": "C", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ALLOW_PROTOCOL": "", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
    }
    assert len(git.calls) == 5 and len(git.archive_calls) == 1
    for command, kwargs in [*git.calls, *git.archive_calls]:
        assert kwargs["env"] == expected_environment
        assert "core.hooksPath=/dev/null" in command
        assert "core.attributesFile=/dev/null" in command
        assert "core.fsmonitor=false" in command
        assert str(repo) == command[command.index("-C") + 1]
        assert not kwargs.get("shell", False)
    for _, kwargs in git.calls:
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"]
    assert git.archive_calls[0][1]["stdout"] == subprocess.PIPE
    assert git.archive_calls[0][1]["stderr"] == subprocess.DEVNULL


def test_gzip_is_deterministic_for_same_mock_git_tar(cli, repo, git, tmp_path):
    a = cli.package(repo, tmp_path / "a")
    b = cli.package(repo, tmp_path / "b")
    assert a["sha256"] == b["sha256"] and a["md5"] == b["md5"]


@pytest.mark.parametrize("environment", [None, "jetson", "host", ""])
def test_cpu_docker_marker_required_before_any_git(cli, repo, git, tmp_path, monkeypatch, environment):
    if environment is None:
        monkeypatch.delenv("DEEP_ANC_CONTAINER", raising=False)
    else:
        monkeypatch.setenv("DEEP_ANC_CONTAINER", environment)
    with pytest.raises(ValueError, match="CPU Docker"):
        cli.package(repo, tmp_path / "new")
    assert not git.calls and not (tmp_path / "new").exists()


def test_docker_file_required_before_any_git(cli, repo, git, tmp_path, monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if str(p) == "/.dockerenv" else original(p))
    with pytest.raises(ValueError, match="CPU Docker"):
        cli.package(repo, tmp_path / "new")
    assert not git.calls


@pytest.mark.parametrize("kind", ["repo_link", "parent_link", "parent_move", "linked_gitdir", "git_link",
                                 "objects_link", "nested_link", "config_link", "alternates", "http_alternates",
                                 "commondir", "gitdir", "local_attributes", "promisor"])
def test_non_independent_repository_refused_before_git(cli, repo, git, tmp_path, kind):
    source = repo
    if kind == "repo_link":
        source = tmp_path / "linked_repo"
        source.symlink_to(repo, target_is_directory=True)
    elif kind == "parent_link":
        link = tmp_path / "linked_parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        source = link / repo.name
    elif kind == "parent_move":
        source = repo / "x" / ".."
    elif kind in {"linked_gitdir", "git_link"}:
        saved = tmp_path / "saved_git"
        (repo / ".git").rename(saved)
        if kind == "git_link":
            (repo / ".git").symlink_to(saved, target_is_directory=True)
        else:
            (repo / ".git").write_text(f"gitdir: {saved}\n")
    elif kind in {"objects_link", "config_link"}:
        name = "objects" if kind == "objects_link" else "config"
        saved = tmp_path / f"saved_{name}"
        (repo / ".git" / name).rename(saved)
        (repo / ".git" / name).symlink_to(saved, target_is_directory=saved.is_dir())
    elif kind == "nested_link":
        (repo / ".git/objects/pack/dangling_link").symlink_to(tmp_path / "missing")
    else:
        names = {"alternates": "objects/info/alternates", "http_alternates": "objects/info/http-alternates",
                 "commondir": "commondir", "gitdir": "gitdir", "local_attributes": "info/attributes",
                 "promisor": "objects/pack/fixture.promisor"}
        path = repo / ".git" / names[kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    with pytest.raises(ValueError):
        cli.package(source, tmp_path / "new")
    assert not git.calls and not git.archive_calls and not (tmp_path / "new").exists()
    assert (repo / "untracked_secret_fixture").read_text() == SECRET


@pytest.mark.parametrize("text", ["[include]\npath=/fixture\n", "[includeIf \"gitdir:fixture\"]\npath=x\n",
                                 "[extensions]\npartialClone=x\n", "[remote \"origin\"]\npromisor=true\n",
                                 "[extensions]\nworktreeConfig=true\n", "[core]\nvalue=line\\\nnext\n",
                                 "[core]\nvalue=\0bad\n", "x" * (1024 * 1024 + 1)])
def test_unsafe_local_config_text_refused_before_first_git(cli, repo, git, tmp_path, text):
    (repo / ".git/config").write_text(text)
    with pytest.raises(ValueError):
        cli.package(repo, tmp_path / "new")
    assert not git.calls and not (tmp_path / "new").exists()


@pytest.mark.parametrize("key", ["include.path", "includeIf.gitdir:x.path", "extensions.partialClone",
                                "remote.origin.promisor", "extensions.worktreeConfig"])
def test_git_parsed_config_is_also_checked(cli, repo, git, tmp_path, key):
    git.local_config = key + "\nfixture\0"
    with pytest.raises(ValueError):
        cli.package(repo, tmp_path / "new")
    assert len(git.calls) == 1 and not git.archive_calls and not (tmp_path / "new").exists()


@pytest.mark.parametrize("kind", ["url", "commit", "fsck", "tree", "config_failure"])
def test_source_verification_failures_create_no_output(cli, repo, git, tmp_path, kind):
    if kind == "url":
        git.remote = "https://example.invalid/ESC-50.git"
    elif kind == "commit":
        git.head = "b" * 40
    elif kind == "tree":
        git.tree = "bad-tree"
    else:
        git.fail_action = (("fsck", "--full", "--strict") if kind == "fsck" else
                           ("config", "--local", "--no-includes", "--null", "--list"))
    before = snapshot(repo)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        cli.package(repo, tmp_path / "new")
    assert not (tmp_path / "new").exists() and not git.archive_calls
    assert snapshot(repo) == before


@pytest.mark.parametrize("kind", ["existing", "broken_link", "ancestor_link", "parent_move", "inside_repo", "broad"])
def test_destination_refusal_never_overwrites_or_modifies_source(cli, repo, git, tmp_path, kind):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    out = {"existing": existing, "broken_link": broken, "ancestor_link": link / "new",
           "parent_move": tmp_path / "x" / ".." / "new", "inside_repo": repo / "output",
           "broad": Path("/tmp")}[kind]
    before = snapshot(repo)
    with pytest.raises((ValueError, FileExistsError)):
        cli.package(repo, out)
    assert not git.calls and not git.archive_calls
    assert (existing / "keep").read_text() == "preserve" and snapshot(repo) == before


@pytest.mark.parametrize("kind", ["archive_nonzero", "stream_failure", "empty_stream", "fsync_failure"])
def test_archive_failure_preserves_partial_file_without_success_receipt(cli, repo, git, tmp_path, monkeypatch, kind):
    if kind == "archive_nonzero":
        git.returncode = 1
    elif kind == "stream_failure":
        git.fail_read = True
    elif kind == "empty_stream":
        git.tar = b""
    else:
        def fail(_):
            raise OSError("fixture fsync failure")
        monkeypatch.setattr(cli.os, "fsync", fail)
    before = snapshot(repo)
    out = tmp_path / "partial"
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
        cli.package(repo, out)
    assert out.is_dir() and not (out / "receipt.json").exists()
    files = list(out.iterdir())
    assert len(files) == 1 and files[0].name == f"ESC-50-{cli.COMMIT}.tar.gz"
    assert files[0].stat().st_size > 0 and git.process.killed
    assert snapshot(repo) == before


def test_archive_process_start_failure_keeps_empty_exclusive_file(cli, repo, git, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fixture process start failure")
    monkeypatch.setattr(cli.subprocess, "Popen", fail)
    before = snapshot(repo)
    out = tmp_path / "partial"
    with pytest.raises(OSError, match="process start"):
        cli.package(repo, out)
    files = list(out.iterdir())
    assert len(files) == 1 and files[0].stat().st_size == 0
    assert files[0].name == f"ESC-50-{cli.COMMIT}.tar.gz"
    assert not (out / "receipt.json").exists() and snapshot(repo) == before


def test_process_cleanup_error_does_not_mask_original_failure(cli, repo, git, tmp_path, monkeypatch):
    def kill_failure(self):
        raise OSError("fixture cleanup failure")
    monkeypatch.setattr(FakeProcess, "kill", kill_failure)
    git.fail_read = True
    out = tmp_path / "partial"
    with pytest.raises(OSError, match="interrupted stream"):
        cli.package(repo, out)
    assert len(list(out.iterdir())) == 1 and not (out / "receipt.json").exists()


def test_cli_failure_hides_subprocess_output_and_emits_no_receipt(cli, repo, git, tmp_path, monkeypatch, capsys):
    git.fail_action = ("fsck", "--full", "--strict")
    monkeypatch.setattr(cli.sys, "argv", ["package_esc50_git.py", "--source-repo", str(repo), "--out", str(tmp_path / "new")])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "[실패]" in captured.err
    assert SECRET not in captured.err and not (tmp_path / "new").exists()


def test_cli_success_and_second_call_refuses_existing_output(cli, repo, git, tmp_path, monkeypatch, capsys):
    out = tmp_path / "new"
    monkeypatch.setattr(cli.sys, "argv", ["package_esc50_git.py", "--source-repo", str(repo), "--out", str(out)])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["source_checksum_verified"] is False
    before = snapshot(out)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "[실패]" in captured.err
    assert snapshot(out) == before
