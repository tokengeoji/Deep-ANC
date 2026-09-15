"""실제 다운로드 없이 mocked 공식 GET으로 임시 staging의 쓰기/검증 경계를 시험한다."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from deep_anc.data import archive_staging as staging


PAYLOAD = b"official archive fixture\x00" * 31
URL = "https://www.openslr.org/resources/12/fixture.tar.gz"


def source(algorithm="sha256"):
    return {"name": "fixture.tar.gz", "url": URL, "checksum_algorithm": algorithm,
            "checksum": hashlib.new(algorithm, PAYLOAD).hexdigest(),
            "license": "CC BY 4.0", "official_reference": "https://www.openslr.org/12"}


class Response(io.BytesIO):
    def __init__(self, payload=PAYLOAD, *, headers=None, status=200, url=URL, fragment=None, fail_after=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if headers is None else headers
        self.status, self.url = status, url
        self.fragment, self.fail_after = fragment, fail_after
        self.read_sizes = []

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_sizes.append(size)
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise OSError("mock network interruption")
        return super().read(min(size, self.fragment) if self.fragment else size)


@pytest.fixture(autouse=True)
def cpu_and_no_network(monkeypatch):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    monkeypatch.setattr(staging, "_open_source", lambda *a, **k: pytest.fail("실제 source GET 호출 금지"))


def invoke(out, *, response=None, metadata=None, **kwargs):
    response = response or Response()
    requests = []
    def opener(request, *, timeout):
        requests.append(request)
        assert request.get_method() == "GET"
        assert request.get_header("Accept-encoding") == "identity"
        assert all("authorization" not in key.lower() for key in request.headers)
        assert not request.unredirected_hdrs and timeout == 120
        return response
    receipt = staging.stage_public_archive(metadata or source(), out,
                                           confirm_temporary_local_staging=True,
                                           opener=opener, **kwargs)
    return receipt, requests


@pytest.mark.parametrize("algorithm", ["md5", "sha1", "sha256"])
def test_bounded_short_reads_verify_official_hash_and_receipt(tmp_path, algorithm):
    response = Response(fragment=7)
    receipt, requests = invoke(tmp_path / "new", response=response, metadata=source(algorithm), chunk_bytes=19)
    assert len(requests) == 1
    assert max(response.read_sizes) == 19 and min(response.read_sizes) > 0
    assert (tmp_path / "new" / "fixture.tar.gz").read_bytes() == PAYLOAD
    assert json.loads((tmp_path / "new" / "receipt.json").read_text()) == receipt
    assert receipt["bytes"] == receipt["size"] == len(PAYLOAD)
    assert receipt["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert receipt["md5"] == hashlib.md5(PAYLOAD).hexdigest()
    assert receipt["source_checksum_verified"] and receipt["local_raw_written"]
    assert not receipt["drive_upload_verified"] and not receipt["local_deletion_performed"]
    assert receipt["delete_after_verified_upload_required"] and receipt["license"] == "CC BY 4.0"


def test_missing_content_length_uses_eof_and_official_hash(tmp_path):
    receipt, _ = invoke(tmp_path / "new", response=Response(headers={}), chunk_bytes=17)
    assert receipt["content_length"] is None and receipt["bytes"] == len(PAYLOAD)


@pytest.mark.parametrize("kind", ["checksum", "shorter", "longer", "interrupted", "empty"])
def test_failed_download_is_preserved_without_receipt(tmp_path, kind):
    metadata = source()
    if kind == "checksum":
        metadata["checksum"] = "0" * 64
        response = Response()
    elif kind == "shorter":
        response = Response(headers={"Content-Length": str(len(PAYLOAD) + 1)})
    elif kind == "longer":
        response = Response(headers={"Content-Length": "1"})
    elif kind == "interrupted":
        response = Response(fragment=7, fail_after=14)
    else:
        response = Response(payload=b"", headers={})
    out = tmp_path / "failed"
    with pytest.raises((ValueError, OSError)):
        invoke(out, metadata=metadata, response=response, chunk_bytes=19)
    assert (out / "fixture.tar.gz").exists()
    assert not (out / "receipt.json").exists()
    if kind == "interrupted":
        assert (out / "fixture.tar.gz").read_bytes() == PAYLOAD[:14]


@pytest.mark.parametrize("response", [
    Response(status=206), Response(headers={"Content-Encoding": "gzip"}),
    Response(headers={"Content-Length": "broken"}), Response(headers={"Content-Length": "0"}),
    Response(headers={"Content-Range": "bytes 0-10/20"}),
    Response(url="https://untrusted.example/archive.zip"),
])
def test_invalid_response_is_rejected_before_local_creation(tmp_path, response):
    out = tmp_path / "new"
    with pytest.raises(ValueError):
        invoke(out, response=response)
    assert not out.exists() and not response.read_sizes


@pytest.mark.parametrize("environment", ["jetson", "approved-transfer", "", "unknown"])
def test_cpu_only_environment_guard_precedes_get(tmp_path, monkeypatch, environment):
    monkeypatch.setenv("DEEP_ANC_CONTAINER", environment)
    with pytest.raises(ValueError, match="CPU Docker"):
        staging.stage_public_archive(source(), tmp_path / "new", confirm_temporary_local_staging=True)
    assert not (tmp_path / "new").exists()


def test_explicit_confirmation_required(tmp_path):
    with pytest.raises(ValueError, match="confirm-temporary"):
        staging.stage_public_archive(source(), tmp_path / "new")


@pytest.mark.parametrize("kind", ["existing", "broken", "ancestor", "parent", "root", "tmp"])
def test_unsafe_outputs_rejected_before_get(tmp_path, kind):
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
        staging.stage_public_archive(source(), out, confirm_temporary_local_staging=True)
    assert (existing / "keep").read_text() == "keep"


@pytest.mark.parametrize("metadata", [
    source() | {"url": "http://www.openslr.org/file"},
    source() | {"url": "https://user:password@www.openslr.org/file"},
    source() | {"name": "../file.zip"}, source() | {"name": "receipt.json"},
    source() | {"checksum": "bad"},
])
def test_invalid_source_rejected_before_get(tmp_path, metadata):
    with pytest.raises(ValueError):
        staging.stage_public_archive(metadata, tmp_path / "new", confirm_temporary_local_staging=True)


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 64 * 1024 * 1024 + 1])
def test_memory_limit_validation(tmp_path, chunk_bytes):
    with pytest.raises(ValueError, match="chunk_bytes"):
        staging.stage_public_archive(source(), tmp_path / "new", confirm_temporary_local_staging=True, chunk_bytes=chunk_bytes)


def test_cli_prints_receipt_only_on_success(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/data/stage_public_archive.py"
    spec = importlib.util.spec_from_file_location("archive_stage_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "research_use": "noncommercial_academic",
                                   "sources": {"fixture": source()}}))
    args = ["--catalog", str(catalog), "--source", "fixture", "--out", str(tmp_path / "new")]
    assert cli.main(args) == 1
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(staging, "_open_source", lambda *a, **k: Response())
    assert cli.main(args + ["--confirm-temporary-local-staging"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source_id"] == "fixture"
    assert output["catalog_sha256"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert cli.main(args + ["--confirm-temporary-local-staging"]) == 1
    assert capsys.readouterr().out == ""
