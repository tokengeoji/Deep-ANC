"""원본 없이 Drive 메타데이터/메모리 전송과 PC bootstrap을 검증한다."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deep_anc.data.drive_inventory import analyze_drive_inventory
from deep_anc.data.drive_transfer import stream_archive_to_drive, validate_source

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    return {"schema_version": 1, "storage_policy": "drive_only_no_local_raw",
            "research_use": "noncommercial_academic", "files": [
                {"id": f"part{i}", "name": f"backup.tar.part-{i:04d}-of-0003",
                 "size": 10, "mime_type": "application/octet-stream"} for i in (0, 2)],
            "multipart_archive": {"expected_parts": 3, "expected_total_bytes": 30,
                                  "prefix": "backup.tar"}}


def test_inventory_missing_parts_never_becomes_training_ready():
    report = analyze_drive_inventory(inventory())
    assert report["archive"]["unconfirmed_part_indices"] == [1]
    assert report["archive"]["observed_bytes"] == 20
    assert not report["training_ready"]
    assert report["raw_pcm_qa_current"] == "not_performed"


def test_matching_size_is_not_hash_or_restore_proof():
    data = inventory()
    data["files"].append({"id": "part1", "name": "backup.tar.part-0001-of-0003",
                          "size": 10, "mime_type": "application/octet-stream"})
    result = analyze_drive_inventory(data)
    assert result["archive"]["size_and_count_match"]
    assert not result["archive"]["restoration_ready"]
    assert not result["archive"]["content_hash_verified"]


@pytest.mark.parametrize("key,value", [("schema_version", 8), ("storage_policy", "local"),
                                     ("research_use", "commercial"), ("files", None)])
def test_inventory_rejects_invalid_contract(key, value):
    data = inventory(); data[key] = value
    with pytest.raises(ValueError):
        analyze_drive_inventory(data)


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_part", "wrong_total", "negative_size", "bool_size", "bad_id", "unknown_family"])
def test_inventory_rejects_ambiguous_files(change):
    data = inventory()
    if change == "duplicate_id":
        data["files"].append(copy.deepcopy(data["files"][0]))
    elif change == "duplicate_part":
        data["files"].append({**data["files"][0], "id": "different"})
    elif change == "wrong_total":
        data["files"][0]["name"] = "backup.tar.part-0000-of-0004"
    elif change in ("negative_size", "bool_size"):
        data["files"][0]["size"] = -1 if change == "negative_size" else True
    elif change == "bad_id":
        data["files"][0]["id"] = "https://not-an-id"
    else:
        data["source_families"] = [{"name": "invented"}]
    with pytest.raises(ValueError):
        analyze_drive_inventory(data)


def source_for(data):
    return {"name": "fixture.zip", "url": "https://zenodo.org/records/1/files/fixture.zip",
            "checksum_algorithm": "sha256", "checksum": hashlib.sha256(data).hexdigest(),
            "license": "test fixture"}


class Source(io.BytesIO):
    def __init__(self, data, *, total=None, url=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if total is None else total)}
        self.url = url or "https://zenodo.org/records/1/files/fixture.zip"
    def geturl(self):
        return self.url


class FakeAPI:
    def __init__(self):
        self.calls, self.payload, self.corrupt = [], bytearray(), False
    def check_destination(self, folder, name):
        self.calls.append("destination")
    def initiate(self, source, folder, total):
        self.calls.append("initiate"); self.name = source["name"]; self.folder = folder
        return "fake-session"
    def put(self, session, chunk, start, total):
        self.calls.append((start, len(chunk))); self.payload.extend(chunk)
        return (200, {}, {"id": "newfile"}) if len(self.payload) == total else (
            308, {"Range": f"bytes=0-{len(self.payload)-1}"}, {})
    def metadata(self, file_id):
        return {"size": len(self.payload), "name": self.name, "parents": [self.folder],
                "md5Checksum": "bad" if self.corrupt else hashlib.md5(self.payload).hexdigest()}


def test_upload_streams_chunks_without_disk(tmp_path):
    data = b"abc" * 200000
    api = FakeAPI()
    report = stream_archive_to_drive(source_for(data), "folder", api,
                                    opener=lambda *a, **kw: Source(data), chunk_bytes=262144)
    assert api.payload == data
    assert report["source_checksum_verified"] and report["drive_checksum_verified"]
    assert not report["local_raw_written"] and not report["pcm_qa_verified"]
    assert list(tmp_path.iterdir()) == []


def test_wrong_source_checksum_never_sends_final_chunk():
    data = b"abcd" * 100000
    source = source_for(data); source["checksum"] = "0" * 64
    api = FakeAPI()
    with pytest.raises(ValueError, match="체크섬"):
        stream_archive_to_drive(source, "folder", api, opener=lambda *a, **kw: Source(data), chunk_bytes=262144)
    assert len(api.payload) == 262144


@pytest.mark.parametrize("actual_delta", [-10, 10])
def test_source_length_mismatch_stops_before_completion(actual_delta):
    data = b"x" * 100
    api = FakeAPI()
    with pytest.raises(ValueError):
        stream_archive_to_drive(source_for(data), "folder", api,
                                opener=lambda *a, **kw: Source(data, total=100 + actual_delta))
    assert not api.payload


def test_existing_destination_is_not_overwritten_or_downloaded():
    api = FakeAPI()
    def exists(*args):
        raise FileExistsError("existing")
    api.check_destination = exists
    with pytest.raises(FileExistsError):
        stream_archive_to_drive(source_for(b"x"), "folder", api,
                                opener=lambda *a, **kw: pytest.fail("network opened"))


def test_wrong_drive_hash_is_not_success():
    api = FakeAPI(); api.corrupt = True
    with pytest.raises(ValueError, match="검증 실패"):
        stream_archive_to_drive(source_for(b"hello"), "folder", api, opener=lambda *a, **kw: Source(b"hello"))


def test_untrusted_redirect_is_rejected():
    api = FakeAPI()
    with pytest.raises(ValueError, match="redirect"):
        stream_archive_to_drive(source_for(b"x"), "folder", api,
                                opener=lambda *a, **kw: Source(b"x", url="https://untrusted.example/x"))
    assert api.calls == ["destination"]


@pytest.mark.parametrize("url", ["http://zenodo.org/a", "https://zenodo.org:8443/a",
                                  "https://user@zenodo.org/a", "https://zenodo.org/a#fragment",
                                  "https://untrusted.example/a"])
def test_every_source_redirect_is_checked_before_request(url):
    from urllib.request import Request
    from deep_anc.data.drive_transfer import _ValidatedSourceRedirect
    handler = _ValidatedSourceRedirect()
    with pytest.raises(ValueError):
        handler.redirect_request(Request("https://zenodo.org/a"), None, 302, "", {}, url)
    api = FakeAPI()
    with pytest.raises(ValueError):
        stream_archive_to_drive(source_for(b"x"), "folder", api,
                                opener=lambda *a, **kw: Source(b"x", url=url))
    assert api.calls == ["destination"]


def test_official_source_redirect_is_allowed():
    from urllib.request import Request
    from deep_anc.data.drive_transfer import _ValidatedSourceRedirect
    request = _ValidatedSourceRedirect().redirect_request(
        Request("https://openslr.org/a"), None, 302, "", {}, "https://us.openslr.org/b")
    assert request.full_url == "https://us.openslr.org/b"


def test_inventory_family_requires_distinct_folder_and_manifest():
    data = inventory()
    data["files"].extend([
        {"id": "raw", "name": "speech", "size": None, "mime_type": "application/vnd.google-apps.folder"},
        {"id": "manifest", "name": "speech.jsonl", "size": 50, "mime_type": "application/jsonl"}])
    data["source_families"] = [{"name": "speech", "raw_folder_id": "raw", "manifest_file_id": "manifest"}]
    assert analyze_drive_inventory(data)["source_families_observed"] == ["speech"]
    for raw_id, manifest_id in [("part0", "part0"), ("raw", "raw"), ("manifest", "raw"), ("raw", "part0")]:
        invalid = copy.deepcopy(data)
        invalid["source_families"][0].update(raw_folder_id=raw_id, manifest_file_id=manifest_id)
        with pytest.raises(ValueError):
            analyze_drive_inventory(invalid)


@pytest.mark.parametrize("archive", [[], "archive", 3])
def test_inventory_rejects_nonobject_archive(archive):
    data = inventory(); data["multipart_archive"] = archive
    with pytest.raises(ValueError):
        analyze_drive_inventory(data)


def test_catalog_checksums_are_valid():
    catalog = json.loads((ROOT / "configs/public_archive_sources.json").read_text())
    for source in catalog["sources"].values():
        validate_source(source)


def test_cli_defaults_to_plan_and_forbids_current_pc_execute():
    args = [sys.executable, str(ROOT / "scripts/data/stream_public_to_drive.py"),
            "--catalog", str(ROOT / "configs/public_archive_sources.json"), "--source", "fma_small"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)["action"] == "plan_only"
    result = subprocess.run(args + ["--execute", "--confirm-source-absent", "--parent-folder-id", "folder"],
                            capture_output=True, text=True)
    assert result.returncode == 1


@pytest.mark.parametrize("mode", [None, "cpu", "jetson", "unknown"])
def test_execute_default_denied_before_auth_import(mode, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("transfer_cli", ROOT / "scripts/data/stream_public_to_drive.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if mode is None:
        monkeypatch.delenv("DEEP_ANC_CONTAINER", raising=False)
    else:
        monkeypatch.setenv("DEEP_ANC_CONTAINER", mode)
    monkeypatch.setattr(sys, "argv", ["cli", "--catalog", str(ROOT / "configs/public_archive_sources.json"),
                                     "--source", "fma_small", "--execute", "--confirm-source-absent",
                                     "--parent-folder-id", "folder"])
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("google.auth"):
            pytest.fail("승인 환경 외에서 인증에 접근")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert module.main() == 1
    assert "ValueError" in capsys.readouterr().err


def bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_acoustic", ROOT / "scripts/data/bootstrap_acoustic.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_bootstrap_creates_synthetic_rir_without_raw(tmp_path, monkeypatch):
    module = bootstrap_module()
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    out = tmp_path / "prepared"
    report = module.build_preparation(out, ROOT / "configs/duct.yaml", None, n_rirs=20)
    assert report["pc_bootstrap_complete"] and not report["dataset_training_ready"]
    assert report["storage_policy"] == "drive_archive_with_temporary_local_staging"
    assert report["operation_scope"] == "bootstrap_without_raw_download"
    assert json.loads((out / "preparation.json").read_text()) == report
    assert {path.name for path in out.iterdir()} == {"duct_rirs_v1.npz", "preparation.json", "README.md"}
    with np.load(out / "duct_rirs_v1.npz") as bank:
        assert bank["p_ref"].shape == (20, 8192)
    assert not report["raw_downloaded"] and not report["gpu_training_started"]
    with pytest.raises(FileExistsError):
        module.build_preparation(out, ROOT / "configs/duct.yaml", None, n_rirs=20)


def test_bootstrap_rejects_host_and_symlink(tmp_path, monkeypatch):
    module = bootstrap_module()
    monkeypatch.delenv("DEEP_ANC_CONTAINER", raising=False)
    with pytest.raises(ValueError, match="컨테이너"):
        module.build_preparation(tmp_path / "out", ROOT / "configs/duct.yaml", None)
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="링크"):
        module.build_preparation(tmp_path / "link/out", ROOT / "configs/duct.yaml", None)
