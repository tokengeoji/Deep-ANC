"""실제 네트워크/원본 없이 합성 HTTP 응답만으로 병렬 staging을 검증한다."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import threading

import pytest

from deep_anc.data import parallel_archive_staging as staging


PAYLOAD = b"official parallel archive fixture\x00" * 37
URL = "https://www.openslr.org/resources/12/fixture.tar.gz"


def source(algorithm="sha256", payload=PAYLOAD):
    return {"name": "fixture.tar.gz", "url": URL, "checksum_algorithm": algorithm,
            "checksum": hashlib.new(algorithm, payload).hexdigest(),
            "license": "CC BY 4.0", "official_reference": "https://www.openslr.org/12"}


class Response(io.BytesIO):
    def __init__(self, payload, *, headers, status=206, url=URL, fragment=7, fail_after=None,
                 barrier=None, finish=None):
        super().__init__(payload)
        self.headers, self.status, self.url = headers, status, url
        self.fragment, self.fail_after = fragment, fail_after
        self.read_sizes = []
        self.barrier, self.finish = barrier, finish

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_sizes.append(size)
        if self.barrier is not None:
            barrier, self.barrier = self.barrier, None
            barrier.wait(timeout=10)
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise OSError("mock network interruption")
        return super().read(min(size, self.fragment))

    def __exit__(self, *args):
        if self.finish is not None:
            self.finish()
        return super().__exit__(*args)


class FakeSource:
    def __init__(self, *, payload=PAYLOAD, etag='"stable-v1"', mutate=None, synchronized=False):
        self.payload, self.etag, self.mutate = payload, etag, mutate
        self.requests, self.responses = [], []
        self.lock = threading.Lock()
        self.active = self.max_active = 0
        self.barrier = threading.Barrier(4) if synchronized else None

    def __call__(self, request, *, timeout):
        assert request.get_method() == "GET" and timeout == 120
        assert request.get_header("Accept-encoding") == "identity"
        assert all("authorization" not in key.lower() for key in request.headers)
        assert not request.unredirected_hdrs
        with self.lock:
            probe = not self.requests
            self.requests.append(request)
            if not probe:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match
        start, end = map(int, match.groups())
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                   "Content-Length": str(end - start + 1), "Accept-Ranges": "bytes"}
        if self.etag is not None:
            headers["ETag"] = self.etag
        assert request.get_header("If-range") == (
            self.etag if not probe and self.etag and not self.etag.startswith("W/") else None)
        kwargs = {"headers": headers, "payload": self.payload[start:end + 1]}
        if self.mutate is not None:
            self.mutate(probe, start, end, kwargs)
        def finished():
            with self.lock:
                self.active -= 1
        response = Response(**kwargs, barrier=None if probe else self.barrier,
                            finish=None if probe else finished)
        with self.lock:
            self.responses.append(response)
        return response


@pytest.fixture(autouse=True)
def cpu_and_no_network(monkeypatch):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    monkeypatch.setattr(staging, "_open_source", lambda *a, **k: pytest.fail("실제 source GET 금지"))


def invoke(out, *, fake=None, metadata=None, **kwargs):
    fake = fake or FakeSource()
    receipt = staging.stage_public_archive_parallel(
        metadata or source(), out, confirm_temporary_local_staging=True,
        opener=fake, chunk_bytes=19, **kwargs)
    return receipt, fake


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "md5"])
def test_parallel_ranges_bounded_reads_official_hash_and_compatible_receipt(tmp_path, algorithm):
    out = tmp_path / "new"
    receipt, fake = invoke(out, metadata=source(algorithm), fake=FakeSource(synchronized=True))
    assert len(fake.requests) == 5 and fake.max_active == 4 and fake.active == 0
    assert all(0 < size <= 19 for response in fake.responses for size in response.read_sizes)
    assert (out / "fixture.tar.gz").read_bytes() == PAYLOAD
    parts = receipt["range_parts"]
    assert parts[0]["start"] == 0 and parts[-1]["end_inclusive"] == len(PAYLOAD) - 1
    assert all(a["end_inclusive"] + 1 == b["start"] for a, b in zip(parts, parts[1:]))
    assert b"".join(Path(item["path"]).read_bytes() for item in parts) == PAYLOAD
    assert all(item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() for item in parts)
    assert json.loads((out / "receipt.json").read_text()) == receipt
    assert receipt["bytes"] == receipt["size"] == receipt["content_length"] == len(PAYLOAD)
    assert receipt["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert receipt["md5"] == hashlib.md5(PAYLOAD).hexdigest()
    assert receipt["source_checksum_verified"] and receipt["local_raw_written"]
    assert receipt["range_workers"] == 4 and receipt["range_parts_retained"]
    assert receipt["delete_after_verified_upload_required"]
    assert not receipt["drive_upload_verified"] and not receipt["local_deletion_performed"]
    assert not receipt["pcm_qa_verified"] and not receipt["extraction_performed"]
    assert not receipt["automatic_retry_performed"]


@pytest.mark.parametrize("etag", [None, 'W/"stable-weak"'])
def test_no_or_weak_etag_uses_exact_response_consistency_and_official_hash(tmp_path, etag):
    receipt, fake = invoke(tmp_path / "new", fake=FakeSource(etag=etag))
    assert receipt["source_etag"] == etag
    assert all(request.get_header("If-range") is None for request in fake.requests)


@pytest.mark.parametrize("payload", [b"x", b"xyz"])
def test_small_archive_has_no_empty_or_overlapping_fragments(tmp_path, payload):
    receipt, fake = invoke(tmp_path / "new", fake=FakeSource(payload=payload), metadata=source(payload=payload))
    assert receipt["range_workers"] == len(payload)
    assert len(fake.requests) == 1 + len(payload)
    assert (tmp_path / "new" / "fixture.tar.gz").read_bytes() == payload


@pytest.mark.parametrize("kind", ["ignored", "rate_limit", "wrong_start", "wrong_end", "unknown_total",
                                  "zero_total", "range_unit", "length", "encoding", "no_ranges",
                                  "etag_syntax", "bad_url", "short", "long"])
def test_probe_failure_precedes_local_creation_and_does_not_retry(tmp_path, kind):
    def mutate(probe, start, end, kwargs):
        assert probe
        headers = kwargs["headers"]
        if kind == "ignored": kwargs["status"] = 200
        elif kind == "rate_limit": kwargs["status"] = 429
        elif kind == "wrong_start": headers["Content-Range"] = f"bytes 1-1/{len(PAYLOAD)}"
        elif kind == "wrong_end": headers["Content-Range"] = f"bytes 0-1/{len(PAYLOAD)}"
        elif kind == "unknown_total": headers["Content-Range"] = "bytes 0-0/*"
        elif kind == "zero_total": headers["Content-Range"] = "bytes 0-0/0"
        elif kind == "range_unit": headers["Content-Range"] = f"items 0-0/{len(PAYLOAD)}"
        elif kind == "length": headers["Content-Length"] = "2"
        elif kind == "encoding": headers["Content-Encoding"] = "gzip"
        elif kind == "no_ranges": headers["Accept-Ranges"] = "none"
        elif kind == "etag_syntax": headers["ETag"] = "bare-tag"
        elif kind == "bad_url": kwargs["url"] = "https://untrusted.example/archive"
        elif kind == "short": kwargs["payload"] = b""
        else: kwargs["payload"] = b"xx"
    fake = FakeSource(mutate=mutate)
    with pytest.raises(ValueError):
        invoke(tmp_path / "new", fake=fake)
    assert len(fake.requests) == 1 and not (tmp_path / "new").exists()


@pytest.mark.parametrize("kind", ["ignored", "rate_limit", "total", "start", "end", "encoding",
                                  "etag", "missing_etag", "final_url", "length", "short", "long", "interrupted"])
def test_part_failure_has_no_receipt_no_automatic_retry_and_preserves_partial_files(tmp_path, kind):
    def mutate(probe, start, end, kwargs):
        if probe: return
        headers = kwargs["headers"]
        if kind == "ignored": kwargs["status"] = 200
        elif kind == "rate_limit": kwargs["status"] = 429
        elif kind == "total": headers["Content-Range"] = f"bytes {start}-{end}/{len(PAYLOAD) + 1}"
        elif kind == "start": headers["Content-Range"] = f"bytes {start + 1}-{end}/{len(PAYLOAD)}"
        elif kind == "end": headers["Content-Range"] = f"bytes {start}-{end - 1}/{len(PAYLOAD)}"
        elif kind == "encoding": headers["Content-Encoding"] = "gzip"
        elif kind == "etag": headers["ETag"] = '"changed"'
        elif kind == "missing_etag": headers.pop("ETag")
        elif kind == "final_url": kwargs["url"] = "https://us.openslr.org/changed"
        elif kind == "length": headers["Content-Length"] = "broken"
        elif kind == "short": kwargs["payload"] = kwargs["payload"][:14]
        elif kind == "long": kwargs["payload"] += b"extra"
        else: kwargs["fail_after"] = 14
    out = tmp_path / "failed"
    fake = FakeSource(mutate=mutate)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(out, fake=fake, workers=1)
    assert len(fake.requests) == 2 and (out / "range_parts").is_dir()
    assert not (out / "receipt.json").exists() and not (out / "fixture.tar.gz").exists()
    if kind in {"short", "interrupted", "long"}:
        assert (out / "range_parts" / "part-00.bin").exists()
        expected = PAYLOAD if kind == "long" else PAYLOAD[:14]
        assert (out / "range_parts" / "part-00.bin").read_bytes() == expected


def test_wrong_official_checksum_preserves_full_archive_and_fragments_without_receipt(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="checksum"):
        invoke(out, metadata=source() | {"checksum": "0" * 64})
    assert (out / "fixture.tar.gz").read_bytes() == PAYLOAD
    assert len(list((out / "range_parts").iterdir())) == 4
    assert not (out / "receipt.json").exists()


def test_parallel_failure_stops_without_retry_or_publishing_receipt(tmp_path):
    def mutate(probe, start, end, kwargs):
        if not probe and start == 0:
            kwargs["fail_after"] = 14
    out = tmp_path / "new"
    fake = FakeSource(mutate=mutate, synchronized=True)
    with pytest.raises((OSError, RuntimeError)):
        invoke(out, fake=fake)
    assert len(fake.requests) == 5 and fake.max_active == 4 and fake.active == 0
    ranges = [request.get_header("Range") for request in fake.requests[1:]]
    assert len(set(ranges)) == 4
    assert (out / "range_parts" / "part-00.bin").read_bytes() == PAYLOAD[:14]
    assert not (out / "receipt.json").exists() and not (out / "fixture.tar.gz").exists()


@pytest.mark.parametrize("mutation", ["same_size", "shorter", "longer", "symlink"])
def test_changed_part_before_merge_is_rejected_without_receipt(tmp_path, monkeypatch, mutation):
    original = staging.as_completed
    out = tmp_path / "new"
    def tamper_after_completed(futures):
        yield from original(futures)
        part = out / "range_parts" / "part-00.bin"
        data = part.read_bytes()
        if mutation == "symlink":
            target = tmp_path / "fixture-target"
            target.write_bytes(data)
            part.unlink()
            part.symlink_to(target)
        else:
            part.write_bytes({"same_size": bytes([data[0] ^ 1]) + data[1:],
                              "shorter": data[:-1], "longer": data + b"x"}[mutation])
    monkeypatch.setattr(staging, "as_completed", tamper_after_completed)
    with pytest.raises(ValueError, match="부분 파일"):
        invoke(out)
    assert not (out / "receipt.json").exists()
    assert len(list((out / "range_parts").iterdir())) == 4


@pytest.mark.parametrize("workers", [0, 5, -1, True, 1.5])
def test_worker_bound_is_checked_before_network(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        staging.stage_public_archive_parallel(source(), tmp_path / "new", workers=workers,
                                               confirm_temporary_local_staging=True)


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 64 * 1024 * 1024 + 1])
def test_chunk_bound_is_checked_before_network(tmp_path, chunk_bytes):
    with pytest.raises(ValueError, match="chunk_bytes"):
        staging.stage_public_archive_parallel(source(), tmp_path / "new", chunk_bytes=chunk_bytes,
                                               confirm_temporary_local_staging=True)


@pytest.mark.parametrize("environment", ["jetson", "approved-transfer", "", "unknown"])
def test_cpu_only_guard(tmp_path, monkeypatch, environment):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", environment)
    with pytest.raises(ValueError, match="CPU Docker"):
        staging.stage_public_archive_parallel(source(), tmp_path / "new", confirm_temporary_local_staging=True)


def test_confirmation_is_required_before_network(tmp_path):
    with pytest.raises(ValueError, match="confirm-temporary"):
        staging.stage_public_archive_parallel(source(), tmp_path / "new")


@pytest.mark.parametrize("kind", ["existing", "broken", "ancestor", "parent", "root", "tmp"])
def test_unsafe_output_rejected_before_network_and_existing_data_preserved(tmp_path, kind):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    out = {"existing": existing, "broken": broken, "ancestor": link / "new",
           "parent": tmp_path / "unused" / ".." / "new", "root": Path("/"), "tmp": Path("/tmp")}[kind]
    with pytest.raises((ValueError, FileExistsError)):
        staging.stage_public_archive_parallel(source(), out, confirm_temporary_local_staging=True)
    assert (existing / "keep").read_text() == "keep"


@pytest.mark.parametrize("change", [
    {"url": "http://www.openslr.org/file"}, {"url": "https://user:password@www.openslr.org/file"},
    {"url": "https://www.openslr.org:8443/file"}, {"url": "https://untrusted.example/file"},
    {"name": "../file.zip"}, {"name": "receipt.json"}, {"name": "range_parts"}, {"checksum": "bad"},
])
def test_source_policy_is_not_weakened(tmp_path, change):
    with pytest.raises(ValueError):
        staging.stage_public_archive_parallel(source() | change, tmp_path / "new",
                                               confirm_temporary_local_staging=True)


def test_cli_success_receipt_and_failure_no_stdout(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/data/stage_public_archive_parallel.py"
    spec = importlib.util.spec_from_file_location("parallel_archive_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "research_use": "noncommercial_academic",
                                   "sources": {"fixture": source()}}))
    args = ["--catalog", str(catalog), "--source", "fixture", "--out", str(tmp_path / "new")]
    assert cli.main(args) == 1 and capsys.readouterr().out == ""
    monkeypatch.setattr(staging, "_open_source", FakeSource())
    assert cli.main(args + ["--confirm-temporary-local-staging"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["source_id"] == "fixture"
    assert receipt["catalog_sha256"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert cli.main(args + ["--confirm-temporary-local-staging"]) == 1
    assert capsys.readouterr().out == ""


def test_cli_http_error_does_not_expose_response_body_or_url(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/data/stage_public_archive_parallel.py"
    spec = importlib.util.spec_from_file_location("parallel_archive_cli_error", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "research_use": "noncommercial_academic",
                                   "sources": {"fixture": source()}}))
    def fail(*args, **kwargs):
        raise RuntimeError("private-response-body-and-url")
    monkeypatch.setattr(staging, "_open_source", fail)
    assert cli.main(["--catalog", str(catalog), "--source", "fixture", "--out", str(tmp_path / "new"),
                     "--confirm-temporary-local-staging"]) == 1
    output = capsys.readouterr()
    assert not output.out and "private-response-body-and-url" not in output.err
