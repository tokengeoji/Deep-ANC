"""scripts/elice/pget.py의 재시도와 완전성 검증."""

from __future__ import annotations

import io
import json
import os
import re
import threading

import pytest

from scripts.elice import pget


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        self._body = io.BytesIO(body)
        self.headers = headers
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeUrlopen:
    def __init__(
        self,
        payload: bytes,
        *,
        head_failures: int = 0,
        incomplete_ranges: bool = False,
        incomplete_starts: set[int] | None = None,
        head_etag: str | None = None,
        range_etag: str | None = None,
        last_modified: str | None = None,
        omit_range_etag: bool = False,
        omit_range_last_modified: bool = False,
    ):
        self.payload = payload
        self.head_failures = head_failures
        self.incomplete_ranges = incomplete_ranges
        self.incomplete_starts = set(incomplete_starts or ())
        self.head_etag = head_etag
        self.range_etag = head_etag if range_etag is None else range_etag
        self.last_modified = last_modified
        self.omit_range_etag = omit_range_etag
        self.omit_range_last_modified = omit_range_last_modified
        self.head_calls = 0
        self.if_ranges: list[str | None] = []
        self.range_requests: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        if request.get_method() == "HEAD":
            with self._lock:
                self.head_calls += 1
                call = self.head_calls
            if call <= self.head_failures:
                raise OSError("temporary SSL EOF")
            headers = {"Content-Length": str(len(self.payload))}
            if self.head_etag is not None:
                headers["ETag"] = self.head_etag
            if self.last_modified is not None:
                headers["Last-Modified"] = self.last_modified
            return _Response(b"", headers, status=200)

        value = request.get_header("Range")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", value or "")
        assert match is not None
        request_headers = {key.lower(): value for key, value in request.header_items()}
        with self._lock:
            self.if_ranges.append(request_headers.get("if-range"))
        start, end = (int(part) for part in match.groups())
        with self._lock:
            self.range_requests.append((start, end))
        body = self.payload[start : end + 1]
        if (self.incomplete_ranges or start in self.incomplete_starts) and body:
            body = body[:-1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"}
        if self.range_etag is not None and not self.omit_range_etag:
            headers["ETag"] = self.range_etag
        if self.last_modified is not None and not self.omit_range_last_modified:
            headers["Last-Modified"] = self.last_modified
        return _Response(
            body,
            headers,
            status=206,
        )


class _BlockingHeadUrlopen(_FakeUrlopen):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.entered = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False
        self._block_lock = threading.Lock()

    def __call__(self, request, timeout=None):
        if request.get_method() == "HEAD":
            with self._block_lock:
                should_block = not self._blocked_once
                self._blocked_once = True
            if should_block:
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise TimeoutError("test did not release blocked HEAD")
        return super().__call__(request, timeout=timeout)


def test_success_uses_part_then_atomic_replace(tmp_path):
    payload = bytes(range(251)) * 17
    output = tmp_path / "archive.bin"
    fake = _FakeUrlopen(payload)

    pget.download(
        "https://example.invalid/archive.bin",
        output,
        connections=4,
        opener=fake,
        attempts=2,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert output.read_bytes() == payload
    assert not (tmp_path / "archive.bin.part").exists()
    assert not (tmp_path / "archive.bin.part.state.json").exists()


def test_head_content_length_is_retried(tmp_path):
    payload = b"head retry payload"
    output = tmp_path / "retry.bin"
    fake = _FakeUrlopen(payload, head_failures=1)

    pget.download(
        "https://example.invalid/retry.bin",
        output,
        connections=2,
        opener=fake,
        attempts=3,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert fake.head_calls == 2
    assert output.read_bytes() == payload


def test_incomplete_ranges_make_cli_nonzero_and_leave_no_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "incomplete.bin"
    fake = _FakeUrlopen(b"0123456789abcdef", incomplete_ranges=True)
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)
    monkeypatch.setattr(pget, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(pget, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(pget, "PROGRESS_INTERVAL_SECONDS", 0.01)

    result = pget.main(
        ["https://example.invalid/incomplete.bin", str(output), "2"]
    )

    assert result != 0
    assert not output.exists()
    assert not (tmp_path / "incomplete.bin.part").exists()
    assert not (tmp_path / "incomplete.bin.part.state.json").exists()


def test_same_output_concurrent_cli_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "shared.bin"
    fake = _BlockingHeadUrlopen(b"one writer only")
    first_errors = []

    def first_download():
        try:
            pget.download(
                "https://example.invalid/shared.bin",
                output,
                connections=2,
                opener=fake,
                attempts=1,
                retry_delay=0,
                progress_interval=0.01,
            )
        except Exception as exc:  # pragma: no cover - assertion reports the value
            first_errors.append(exc)

    first = threading.Thread(target=first_download)
    first.start()
    assert fake.entered.wait(timeout=2)
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)

    try:
        result = pget.main(
            ["https://example.invalid/shared.bin", str(output), "2"]
        )
        assert result != 0
        assert not output.exists()
    finally:
        fake.release.set()
        first.join(timeout=5)

    assert not first.is_alive()
    assert first_errors == []
    assert output.read_bytes() == b"one writer only"


def test_unlocked_stale_lock_file_is_safely_reused(tmp_path):
    output = tmp_path / "stale.bin"
    lock = tmp_path / "stale.bin.part.lock"
    lock.touch()

    pget.download(
        "https://example.invalid/stale.bin",
        output,
        connections=1,
        opener=_FakeUrlopen(b"stale lock is only an inode"),
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert output.read_bytes() == b"stale lock is only an inode"
    assert lock.exists()


def test_etag_mismatch_sends_if_range_and_makes_cli_nonzero(
    tmp_path, monkeypatch
):
    output = tmp_path / "changed.bin"
    fake = _FakeUrlopen(
        b"object changed during download",
        head_etag='"version-1"',
        range_etag='"version-2"',
    )
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)
    monkeypatch.setattr(pget, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(pget, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(pget, "PROGRESS_INTERVAL_SECONDS", 0.01)

    result = pget.main(
        ["https://example.invalid/changed.bin", str(output), "1"]
    )

    assert result != 0
    assert fake.if_ranges == ['"version-1"']
    assert not output.exists()
    assert not (tmp_path / "changed.bin.part").exists()
    assert not (tmp_path / "changed.bin.part.state.json").exists()


def test_strong_etag_must_also_be_present_on_every_range_response(tmp_path):
    output = tmp_path / "missing-range-etag.bin"
    fake = _FakeUrlopen(
        b"validator must bind every 206 response",
        head_etag='"version-1"',
        omit_range_etag=True,
    )

    with pytest.raises(pget.DownloadError, match="strong ETag Range 응답 결속 실패"):
        pget.download(
            "https://example.invalid/missing-range-etag.bin",
            output,
            connections=2,
            opener=fake,
            attempts=1,
            retry_delay=0,
            progress_interval=0.01,
        )

    assert not output.exists()
    assert not (tmp_path / "missing-range-etag.bin.part").exists()
    assert not (tmp_path / "missing-range-etag.bin.part.state.json").exists()


def _leave_verified_partial(
    monkeypatch,
    output,
    payload: bytes,
    *,
    url: str = "https://example.invalid/resume.bin",
    etag: str = '"version-1"',
    connections: int = 1,
):
    block_bytes = len(payload) // 4
    monkeypatch.setattr(pget, "RESUME_RANGE_BYTES", block_bytes)
    failed_start = block_bytes * 2
    fake = _FakeUrlopen(
        payload,
        head_etag=etag,
        incomplete_starts={failed_start},
    )
    with pytest.raises(pget.DownloadError):
        pget.download(
            url,
            output,
            connections=connections,
            opener=fake,
            attempts=1,
            retry_delay=0,
            progress_interval=0.01,
        )
    return fake, failed_start


def test_restart_reuses_only_completed_ranges_when_worker_count_changes(
    tmp_path, monkeypatch
):
    payload = bytes(range(100))
    output = tmp_path / "resume.bin"
    url = "https://example.invalid/resume.bin?secret=not-written"
    _first, failed_start = _leave_verified_partial(
        monkeypatch, output, payload, url=url, connections=1
    )
    part = tmp_path / "resume.bin.part"
    state_path = tmp_path / "resume.bin.part.state.json"

    assert part.stat().st_size == len(payload)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == pget.RESUME_SCHEMA_VERSION
    assert "connections" not in state
    assert state["range_bytes"] == len(payload) // 4
    assert state["completed"] == [True, True, False, True]
    assert state["range_sha256"][2] is None
    assert all(
        digest is not None
        for index, digest in enumerate(state["range_sha256"])
        if index != 2
    )
    assert "url" not in state
    assert "secret=not-written" not in state_path.read_text(encoding="utf-8")

    second = _FakeUrlopen(payload, head_etag='"version-1"')
    pget.download(
        url,
        output,
        connections=4,
        opener=second,
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert second.range_requests == [(failed_start, len(payload) * 3 // 4 - 1)]
    assert output.read_bytes() == payload
    assert not part.exists()
    assert not state_path.exists()
    assert not list(tmp_path.glob(".*.quarantine.*"))


def test_last_modified_is_if_range_only_and_never_cross_process_authority(
    tmp_path, monkeypatch
):
    payload = bytes(range(100))
    output = tmp_path / "resume-last-modified.bin"
    url = "https://example.invalid/resume-last-modified.bin"
    last_modified = "Mon, 31 Aug 2026 00:00:00 GMT"
    block_bytes = len(payload) // 4
    monkeypatch.setattr(pget, "RESUME_RANGE_BYTES", block_bytes)
    failed_start = block_bytes * 2
    first = _FakeUrlopen(
        payload,
        head_etag='W/"weak-version"',
        last_modified=last_modified,
        incomplete_starts={failed_start},
    )

    with pytest.raises(pget.DownloadError):
        pget.download(
            url,
            output,
            connections=4,
            opener=first,
            attempts=1,
            retry_delay=0,
            progress_interval=0.01,
        )

    part_path = tmp_path / "resume-last-modified.bin.part"
    state_path = tmp_path / "resume-last-modified.bin.part.state.json"
    assert first.if_ranges == [last_modified] * 4
    assert not part_path.exists()
    assert not state_path.exists()

    second_payload = payload[::-1]
    second = _FakeUrlopen(
        second_payload,
        head_etag='W/"weak-version"',
        last_modified=last_modified,
    )
    pget.download(
        url,
        output,
        connections=4,
        opener=second,
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert sorted(second.range_requests) == pget._range_layout(len(second_payload))
    assert second.if_ranges == [last_modified] * 4
    assert output.read_bytes() == second_payload
    assert not state_path.exists()


def test_default_resume_layout_is_fixed_64_mib_blocks():
    block = 64 << 20

    assert pget.RESUME_RANGE_BYTES == block
    assert pget._range_layout(block * 2 + 1) == [
        (0, block - 1),
        (block, block * 2 - 1),
        (block * 2, block * 2),
    ]


def test_connections_have_a_fail_fast_upper_bound(tmp_path):
    def forbidden_opener(*_args, **_kwargs):
        raise AssertionError("connection validation must run before network access")

    with pytest.raises(pget.DownloadError, match="연결 수 N은 64 이하여야"):
        pget.download(
            "https://example.invalid/too-many-workers.bin",
            tmp_path / "too-many-workers.bin",
            connections=pget.MAX_CONNECTIONS + 1,
            opener=forbidden_opener,
        )

    assert not list(tmp_path.iterdir())


def test_sidecar_cleanup_failure_after_publish_does_not_fail_output(
    tmp_path, monkeypatch, capsys
):
    payload = b"publish is already durable"
    output = tmp_path / "published.bin"
    state_path = tmp_path / "published.bin.part.state.json"
    real_remove = pget._remove_regular

    def fail_only_sidecar(path):
        if path == state_path:
            return False
        return real_remove(path)

    monkeypatch.setattr(pget, "_remove_regular", fail_only_sidecar)
    fake = _FakeUrlopen(payload, head_etag='"version-1"')

    pget.download(
        "https://example.invalid/published.bin",
        output,
        connections=2,
        opener=fake,
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert output.read_bytes() == payload
    assert state_path.is_file()
    assert fake.range_requests == [(0, len(payload) - 1)]
    assert "완성 파일은 정상" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutation",
    [
        "remote",
        "sidecar",
        "part_size",
        "part_bytes",
        "range_layout",
        "url",
        "types",
    ],
)
def test_stale_or_corrupt_resume_state_is_quarantined_before_fresh_download(
    tmp_path, monkeypatch, mutation
):
    payload = bytes(range(100))
    output = tmp_path / f"restart-{mutation}.bin"
    first_url = "https://example.invalid/object-v1.bin"
    _leave_verified_partial(monkeypatch, output, payload, url=first_url)
    part = tmp_path / f"restart-{mutation}.bin.part"
    state_path = tmp_path / f"restart-{mutation}.bin.part.state.json"
    second_payload = payload
    second_etag = '"version-1"'
    second_url = first_url

    if mutation == "remote":
        second_payload = payload[::-1]
        second_etag = '"version-2"'
    elif mutation == "sidecar":
        state_path.write_bytes(b'{"truncated":')
    elif mutation == "part_size":
        with part.open("r+b") as handle:
            handle.truncate(len(payload) - 1)
    elif mutation == "part_bytes":
        with part.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\xff")
    elif mutation == "range_layout":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["ranges"][0][1] -= 1
        state_path.write_text(
            json.dumps(pget._seal_state(state)), encoding="utf-8"
        )
    elif mutation == "url":
        second_url = "https://example.invalid/object-v2.bin"
    elif mutation == "types":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # JSON bool은 Python에서 int와 동등 비교되므로 digest를 다시 봉인한
        # 위조 상태도 strict 타입 검증으로 거부해야 한다.
        state["range_bytes"] = True
        state_path.write_text(
            json.dumps(pget._seal_state(state)), encoding="utf-8"
        )

    second = _FakeUrlopen(second_payload, head_etag=second_etag)
    pget.download(
        second_url,
        output,
        connections=4,
        opener=second,
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    expected_ranges = pget._range_layout(len(second_payload))
    assert sorted(second.range_requests) == expected_ranges
    assert output.read_bytes() == second_payload
    quarantines = list(tmp_path.glob(f".{part.name}.quarantine.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / part.name).is_file()
    assert (quarantines[0] / state_path.name).is_file()
    assert not part.exists()
    assert not state_path.exists()


def test_quarantine_namespace_collision_never_overwrites_existing_entry(
    tmp_path, monkeypatch
):
    payload = bytes(range(100))
    output = tmp_path / "collision.bin"
    url = "https://example.invalid/collision-v1.bin"
    _leave_verified_partial(monkeypatch, output, payload, url=url)
    part = tmp_path / "collision.bin.part"
    preexisting = tmp_path / f".{part.name}.quarantine.preexisting"
    preexisting.mkdir(mode=0o700)
    sentinel = preexisting / "sentinel.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    second = _FakeUrlopen(payload, head_etag='"version-1"')
    pget.download(
        "https://example.invalid/collision-v2.bin",
        output,
        connections=4,
        opener=second,
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    quarantine_dirs = list(tmp_path.glob(f".{part.name}.quarantine.*"))
    assert len(quarantine_dirs) == 2
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    created = next(path for path in quarantine_dirs if path != preexisting)
    assert (created / part.name).is_file()
    assert output.read_bytes() == payload


def test_symlink_partial_is_rejected_without_touching_target(tmp_path):
    payload = b"safe remote bytes"
    output = tmp_path / "symlink.bin"
    target = tmp_path / "target.bin"
    target.write_bytes(b"must remain unchanged")
    part = tmp_path / "symlink.bin.part"
    part.symlink_to(target)

    with pytest.raises(pget.DownloadError, match="symlink/non-regular"):
        pget.download(
            "https://example.invalid/symlink.bin",
            output,
            connections=2,
            opener=_FakeUrlopen(payload, head_etag='"version-1"'),
            attempts=1,
            retry_delay=0,
            progress_interval=0.01,
        )

    assert target.read_bytes() == b"must remain unchanged"
    assert part.is_symlink()
    assert not output.exists()


@pytest.mark.parametrize("kind", ["part", "lock"])
def test_hardlinked_internal_paths_are_rejected_without_touching_target(
    tmp_path, kind
):
    payload = b"safe remote bytes"
    output = tmp_path / f"hardlink-{kind}.bin"
    target = tmp_path / f"hardlink-{kind}-target.bin"
    target.write_bytes(b"must remain unchanged")
    guarded = (
        tmp_path / f"hardlink-{kind}.bin.part"
        if kind == "part"
        else tmp_path / f"hardlink-{kind}.bin.part.lock"
    )
    os.link(target, guarded)

    with pytest.raises(pget.DownloadError, match="ownership/link 계약 불일치"):
        pget.download(
            f"https://example.invalid/hardlink-{kind}.bin",
            output,
            connections=2,
            opener=_FakeUrlopen(payload, head_etag='"version-1"'),
            attempts=1,
            retry_delay=0,
            progress_interval=0.01,
        )

    assert target.read_bytes() == b"must remain unchanged"
    assert guarded.samefile(target)
    assert not output.exists()
