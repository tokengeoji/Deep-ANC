from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.elice import push_transfer_bundle as push


def _key(path: Path, *, mode: int = 0o600) -> Path:
    path.write_bytes(
        b"-----BEGIN "
        + b"OPENSSH PRIVATE KEY-----\n"
        + b"A" * 160
        + b"\n-----END "
        + b"OPENSSH PRIVATE KEY-----\n"
    )
    path.chmod(mode)
    return path


def _endpoint(identity: Path) -> push.Endpoint:
    return push.Endpoint(
        host="central-02.tcp.tunnel.elice.io",
        port=14796,
        user="elicer",
        identity=identity,
        remote_repo="/home/elicer/Deep_ANC",
    )


def _bundle() -> push.LocalBundle:
    manifest_raw = b"manifest"
    return push.LocalBundle(
        commit="a" * 40,
        manifest=push.FileEntry(
            path="data/manifests/elice_transfer_manifest.json",
            sha256=hashlib.sha256(manifest_raw).hexdigest(),
            size=len(manifest_raw),
        ),
        files=(
            push.FileEntry(path="data/recorded/a.wav", sha256="b" * 64, size=12),
            push.FileEntry(path="assets/measured/p.npz", sha256="c" * 64, size=34),
        ),
        schema_version=2,
    )


def test_identity_requires_private_regular_mode_and_rejects_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "keys"
    outside.mkdir()
    valid = _key(outside / "elice.pem")
    assert push._validate_identity(os.fspath(valid), repo_root=repo) == valid.resolve()

    permissive = _key(outside / "permissive.pem", mode=0o644)
    with pytest.raises(push.PushTransferError, match="group/other"):
        push._validate_identity(os.fspath(permissive), repo_root=repo)

    link = outside / "linked.pem"
    link.symlink_to(valid)
    with pytest.raises(push.PushTransferError, match="symlink"):
        push._validate_identity(os.fspath(link), repo_root=repo)

    linked_parent = tmp_path / "linked-keys"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(push.PushTransferError, match="symlink component"):
        push._validate_identity(
            os.fspath(linked_parent / valid.name), repo_root=repo
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "central;touch-x"),
        ("user", "elicer$(id)"),
        ("remote_repo", "/home/elicer/Deep_ANC;touch-x"),
        ("remote_repo", "/home/elicer/../root"),
        ("port", 0),
    ],
)
def test_endpoint_rejects_command_injection_and_unsafe_paths(
    tmp_path: Path, field: str, value: object
) -> None:
    identity = _key(tmp_path / "key.pem")
    values: dict[str, object] = {
        "host": "central-02.tcp.tunnel.elice.io",
        "port": 14796,
        "user": "elicer",
        "identity": os.fspath(identity),
        "remote_repo": "/home/elicer/Deep_ANC",
    }
    values[field] = value
    with pytest.raises(push.PushTransferError):
        push._endpoint_from_args(argparse.Namespace(**values), repo_root=tmp_path / "repo")


def test_rsync_is_relative_partial_from0_without_delete_and_includes_manifest(
    tmp_path: Path,
) -> None:
    endpoint = _endpoint(tmp_path / "private.pem")
    bundle = _bundle()
    command = push._rsync_command(endpoint)
    assert "-aR" in command
    assert "--partial" in command
    assert "--from0" in command
    assert "--files-from=-" in command
    assert not any("delete" in argument for argument in command)
    paths = push._file_list_bytes(bundle).rstrip(b"\0").decode().split("\0")
    assert paths[-1] == "data/manifests/elice_transfer_manifest.json"
    assert set(paths) == set(bundle.transfer_paths)


def test_remote_payload_never_contains_identity_path_or_key_material(tmp_path: Path) -> None:
    identity = tmp_path / "do-not-log-this-private.pem"
    endpoint = _endpoint(identity)
    payload = push._remote_payload(_bundle(), endpoint, phase="pre")
    assert os.fsencode(identity) not in payload
    assert b"PRIVATE KEY" not in payload
    decoded = json.loads(payload)
    assert decoded["phase"] == "pre"
    assert decoded["files"][-1]["path"] == "data/manifests/elice_transfer_manifest.json"


def test_dry_run_validates_local_only_and_makes_zero_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _bundle()
    endpoint = _endpoint(tmp_path / "private-secret-name.pem")
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(push, "_endpoint_from_args", lambda _args, repo_root: endpoint)
    monkeypatch.setattr(push, "_validate_local_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(push.shutil, "which", lambda name: f"/usr/bin/{name}")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run에서 network primitive를 호출하면 안 됩니다")

    monkeypatch.setattr(push, "_run_remote_check", forbidden)
    monkeypatch.setattr(push, "_run_rsync", forbidden)
    status = push.main(
        [
            "--host",
            endpoint.host,
            "--port",
            str(endpoint.port),
            "--user",
            endpoint.user,
            "--identity",
            os.fspath(endpoint.identity),
            "--remote-repo",
            endpoint.remote_repo,
            "--expected-commit",
            bundle.commit,
            "--expected-manifest-sha256",
            bundle.manifest.sha256,
            "--dry-run",
        ]
    )
    assert status == 0
    output = capsys.readouterr()
    assert "network_calls=0" in output.out
    assert os.fspath(endpoint.identity) not in output.out + output.err


def test_live_order_is_remote_preflight_rsync_remote_postflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    endpoint = _endpoint(tmp_path / "private.pem")
    calls: list[str] = []
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(push, "_endpoint_from_args", lambda _args, repo_root: endpoint)
    monkeypatch.setattr(push, "_validate_local_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(push.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        push,
        "_run_remote_check",
        lambda _bundle, _endpoint, *, phase: calls.append(phase),
    )
    monkeypatch.setattr(
        push,
        "_run_rsync",
        lambda _bundle, _endpoint, *, repo_root: calls.append("rsync"),
    )
    status = push.main(
        [
            "--host",
            endpoint.host,
            "--port",
            str(endpoint.port),
            "--user",
            endpoint.user,
            "--identity",
            os.fspath(endpoint.identity),
            "--remote-repo",
            endpoint.remote_repo,
            "--expected-commit",
            bundle.commit,
            "--expected-manifest-sha256",
            bundle.manifest.sha256,
        ]
    )
    assert status == 0
    assert calls == ["pre", "rsync", "post"]


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_remote_stdlib_checker_verifies_exact_checkout_and_post_hash(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / ".gitignore").write_text("/data/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("exact\n", encoding="utf-8")
    _git(root, "add", ".gitignore", "tracked.txt")
    _git(root, "commit", "-qm", "fixture")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    raw = b"payload"
    payload = {
        "schema_version": 1,
        "phase": "pre",
        "remote_repo": os.fspath(root),
        "expected_commit": commit,
        "files": [
            {
                "path": "data/value.bin",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        ],
    }

    def check() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", push._REMOTE_CHECK_PROGRAM],
            input=json.dumps(payload).encode(),
            capture_output=True,
            check=False,
        )

    before = check()
    assert before.returncode == 0, before.stderr.decode(errors="replace")
    target = root / "data/value.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    payload["phase"] = "post"
    after = check()
    assert after.returncode == 0, after.stderr.decode(errors="replace")
    assert after.stdout == b"REMOTE_POST_OK\n"

    target.unlink()
    target.symlink_to(root / "tracked.txt")
    rejected = check()
    assert rejected.returncode != 0
    assert b"REMOTE_CHECK_FAIL" in rejected.stderr


def test_inventory_rejects_manifest_self_reference() -> None:
    raw_payload: dict[str, Any] = {
        "schema_version": 1,
        "files": [
            {
                "path": "data/manifests/elice_transfer_manifest.json",
                "sha256": "a" * 64,
                "size": 1,
            }
        ],
    }
    raw = json.dumps(raw_payload).encode()
    with pytest.raises(push.PushTransferError, match="manifest 자체"):
        push._load_inventory(
            raw,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_size=len(raw),
        )


def test_inventory_accepts_stage2_physical_schema_v3_file_list() -> None:
    raw_payload: dict[str, Any] = {
        "schema_version": 3,
        "files": [
            {
                "path": "results/stage2_2khz_ps_v3/plant_binding.json",
                "sha256": "a" * 64,
                "size": 17,
            },
            {
                "path": "authority/stage2_2khz_physical.json",
                "sha256": "b" * 64,
                "size": 19,
            },
        ],
    }
    raw = json.dumps(raw_payload).encode()

    schema, inventory = push._load_inventory(
        raw,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_size=len(raw),
    )

    assert schema == 3
    assert [entry.path for entry in inventory] == [
        "authority/stage2_2khz_physical.json",
        "results/stage2_2khz_ps_v3/plant_binding.json",
    ]


def test_elice_guide_separates_v1_selector_and_v2_training_bootstrap() -> None:
    guide = (push.REPO_ROOT / "docs/05_training_elice.md").read_text(encoding="utf-8")
    cursor = 0
    for token in (
        "v1 selector bootstrap",
        "byte copy",
        "combined 101세션",
        "v2 training bootstrap",
    ):
        position = guide.find(token, cursor)
        assert position >= 0, token
        cursor = position + len(token)
    for token in (
        "scripts/elice/push_transfer_bundle.py",
        "--dry-run",
        "network 0",
        "--reuse-decoder-audit",
        "--cache-preflight-only",
        "receipt를 **0개** 발행",
        "rsync -aR --partial --from0 --files-from=-",
    ):
        assert token in guide
