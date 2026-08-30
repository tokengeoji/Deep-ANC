"""fullband-v5 meter 공용 package 계약의 hermetic 회귀 테스트."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

import numpy as np
import pytest

from deep_anc.dsp import fullband_v5_meter as meter
from deep_anc.dsp import fullband_live_post_v5 as live_post
from deep_anc.dsp.measurement_level import load_measurement_level_evidence
from deep_anc.data import repository_fd
from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    publish_repository_bytes_noreplace,
    repository_execution_identity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _execution() -> dict:
    return {
        "repository_commit": "a" * 40,
        "repository_branch": "work/test",
        "repository_dirty": False,
        "script_path": "scripts/data/set_amp_level.py",
        "script_file_sha256": "b" * 64,
    }


@pytest.fixture(autouse=True)
def _stable_execution_identity(monkeypatch):
    monkeypatch.setattr(meter, "repository_execution_identity", lambda *_a: _execution())


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract() -> dict:
    return {
        "plan": {
            "path": meter.DEFAULT_PLAN_ENVELOPE_PATH,
            "file_sha256": _sha("plan-file"),
            "payload_sha256": _sha("plan-payload"),
            "pcm_sha256": _sha("pcm"),
        },
        "live_capture_authority": {
            "path": meter.DEFAULT_LIVE_AUTHORITY_PATH,
            "file_sha256": _sha("authority-file"),
            "payload_sha256": _sha("authority-payload"),
        },
        "hardware": {
            "path": meter.DEFAULT_HARDWARE_PATH,
            "file_sha256": _sha("hardware-file"),
            "identity_sha256": _sha("hardware-identity"),
            "physical_fingerprint_sha256": _sha("physical"),
        },
        "level_evidence": {
            "path": meter.DEFAULT_LEVEL_EVIDENCE_PATH,
            "file_sha256": _sha("evidence-file"),
            "identity_sha256": _sha("hardware-identity"),
            "scope": meter.TRACKED_V5_LEVEL_ATTESTATION_SCOPE,
            "preserved_raw_revalidated": False,
        },
        "sealed_raw": {
            "path": meter.DEFAULT_RAW_TARGET_PATH,
            "must_not_exist_before_capture": True,
        },
        "hardware_config": {"audio": {}, "channels": {}},
        "hardware_audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {"card": "APE", "pcm": 1},
            "output": {"card": "Audio", "pcm": 0},
        },
        "channel_map": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
        "hardware_identity": {"identity": "fixture"},
        "physical_fingerprint": {"sha256": _sha("physical")},
        "evidence": {"passed": True},
    }


def _confirmations() -> dict[str, bool]:
    return {name: True for name in meter.CONFIRMATION_KEYS}


def _copy_tracked_v5_admission_files(destination: Path) -> None:
    for relative in (
        meter.DEFAULT_PLAN_ENVELOPE_PATH,
        meter.DEFAULT_LIVE_AUTHORITY_PATH,
        meter.DEFAULT_HARDWARE_PATH,
        meter.DEFAULT_LEVEL_EVIDENCE_PATH,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, target)


def test_clean_tracked_only_root_has_portable_attestation_and_static_admission(
    tmp_path: Path,
) -> None:
    _copy_tracked_v5_admission_files(tmp_path)

    attestation = meter.load_tracked_v5_level_attestation(
        tmp_path / meter.DEFAULT_LEVEL_EVIDENCE_PATH,
        repository_root=tmp_path,
    )
    assert attestation["scope"] == meter.TRACKED_V5_LEVEL_ATTESTATION_SCOPE
    assert attestation["preserved_raw_revalidated"] is False
    assert attestation["strict_ps_authority"] is False
    assert attestation["plant_or_training_authority"] is False
    assert attestation["live_admission_eligible"] is False

    # 기존 forensic loader는 그대로 두 raw+meter receipt를 요구한다. clean
    # checkout에서 이 strict 의미를 portable attestation으로 승격하지 않는다.
    with pytest.raises(FileNotFoundError, match="보존 raw"):
        load_measurement_level_evidence(
            tmp_path / meter.DEFAULT_LEVEL_EVIDENCE_PATH,
            repository_root=tmp_path,
        )

    contract = meter.validate_fullband_v5_static_contract(
        repository_root=tmp_path,
        require_sealed_raw_fresh=True,
    )
    assert contract["level_evidence"] == {
        "path": meter.DEFAULT_LEVEL_EVIDENCE_PATH,
        "file_sha256": meter.EXPECTED_TRACKED_LEVEL_EVIDENCE_FILE_SHA256,
        "identity_sha256": attestation["identity_sha256"],
        "scope": meter.TRACKED_V5_LEVEL_ATTESTATION_SCOPE,
        "preserved_raw_revalidated": False,
    }


def test_portable_attestation_tamper_and_symlink_fail_closed(tmp_path: Path) -> None:
    _copy_tracked_v5_admission_files(tmp_path)
    evidence = tmp_path / meter.DEFAULT_LEVEL_EVIDENCE_PATH
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file SHA"):
        meter.load_tracked_v5_level_attestation(
            evidence,
            repository_root=tmp_path,
        )

    evidence.unlink()
    evidence.symlink_to(REPOSITORY_ROOT / meter.DEFAULT_LEVEL_EVIDENCE_PATH)
    with pytest.raises(OSError):
        meter.load_tracked_v5_level_attestation(
            evidence,
            repository_root=tmp_path,
        )


def test_static_admission_rejects_occupied_or_symlinked_raw_target(
    tmp_path: Path,
) -> None:
    _copy_tracked_v5_admission_files(tmp_path)
    raw = tmp_path / meter.DEFAULT_RAW_TARGET_PATH
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        meter.validate_fullband_v5_static_contract(
            repository_root=tmp_path,
            require_sealed_raw_fresh=True,
        )
    raw.unlink()
    raw.symlink_to(tmp_path / meter.DEFAULT_LEVEL_EVIDENCE_PATH)
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        meter.validate_fullband_v5_static_contract(
            repository_root=tmp_path,
            require_sealed_raw_fresh=True,
        )


def test_static_admission_rejects_orphan_external_post_receipt(tmp_path: Path):
    _copy_tracked_v5_admission_files(tmp_path)
    raw = PurePosixPath(meter.DEFAULT_RAW_TARGET_PATH)
    wrong = tmp_path / (raw.with_suffix("").as_posix() + ".post_receipt.json")
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_text("old wrong-name orphan", encoding="utf-8")
    # 과거 잘못 계산한 raw_capture.post_receipt.json은 authoritative sibling이 아니다.
    meter.validate_fullband_v5_static_contract(
        repository_root=tmp_path,
        require_sealed_raw_fresh=True,
    )
    wrong.unlink()
    actual_relative = live_post.external_post_receipt_relative_path(raw.as_posix())
    assert actual_relative.endswith("raw_capture.npz.post_receipt.json")
    receipt = tmp_path / actual_relative
    receipt.write_text("actual orphan", encoding="utf-8")
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        meter.validate_fullband_v5_static_contract(
            repository_root=tmp_path,
            require_sealed_raw_fresh=True,
        )


def test_post_capture_static_mode_requires_existing_regular_sealed_raw(
    tmp_path: Path,
) -> None:
    _copy_tracked_v5_admission_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        meter.validate_fullband_v5_static_contract(
            repository_root=tmp_path,
            require_sealed_raw_fresh=False,
        )


def test_meter_raw_symlink_alias_is_rejected_lexically(tmp_path: Path) -> None:
    real = tmp_path / "real/meter_raw.npz"
    real.parent.mkdir()
    real.write_bytes(b"raw")
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "meter_raw.npz").symlink_to(real)

    with pytest.raises(OSError):
        meter.validate_fullband_v5_meter_raw_static(
            "alias/meter_raw.npz",
            repository_root=tmp_path,
        )
@pytest.mark.parametrize(
    "relative",
    [
        meter.DEFAULT_PLAN_ENVELOPE_PATH,
        meter.DEFAULT_LIVE_AUTHORITY_PATH,
        meter.DEFAULT_HARDWARE_PATH,
    ],
)
def test_actual_static_loader_rejects_pinned_file_tamper(
    tmp_path: Path, relative: str
) -> None:
    _copy_tracked_v5_admission_files(tmp_path)
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        meter.validate_fullband_v5_static_contract(
            repository_root=tmp_path,
            require_sealed_raw_fresh=True,
        )


def test_held_dirfd_guard_detects_parent_rename_symlink_splice(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sealed/parent/evidence.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed bytes")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence.json").write_bytes(b"forged bytes")

    with RepositoryFileGuard(
        tmp_path, "sealed/parent/evidence.json", label="TOCTOU fixture"
    ) as guard:
        saved = tmp_path / "sealed/parent.saved"
        source.parent.rename(saved)
        source.parent.symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="directory inode"):
            guard.verify()

def test_followup_is_exact_and_any_tamper_fails():
    contract = _contract()
    devices = {"input": 3, "output": 4}
    followup = meter.build_fullband_v5_followup(
        contract,
        resolved_devices=devices,
        confirmations=_confirmations(),
    )

    assert meter.validate_fullband_v5_followup(
        followup,
        expected_contract=contract,
        expected_devices=devices,
    ) == followup
    core = {
        key: item for key, item in followup.items() if key != "followup_contract_sha256"
    }
    assert followup["followup_contract_sha256"] == meter.payload_sha256(core)

    tampered = json.loads(json.dumps(followup))
    tampered["resolved_devices"]["output"] = 5
    with pytest.raises(ValueError, match="contract SHA"):
        meter.validate_fullband_v5_followup(
            tampered,
            expected_contract=contract,
            expected_devices=devices,
        )

    extra = json.loads(json.dumps(followup))
    extra["legacy_v4"] = True
    with pytest.raises(ValueError, match="key 집합"):
        meter.validate_fullband_v5_followup(
            extra,
            expected_contract=contract,
            expected_devices=devices,
        )


def test_followup_requires_five_exact_true_confirmations():
    contract = _contract()
    confirmations = _confirmations()
    confirmations["routing_and_geometry"] = False
    with pytest.raises(ValueError, match="다섯"):
        meter.build_fullband_v5_followup(
            contract,
            resolved_devices={"input": 1, "output": 2},
            confirmations=confirmations,
        )


@pytest.mark.parametrize("sealed_fresh", [True, False])
@pytest.mark.parametrize("meter_fresh", [True, False])
def test_saved_meter_consumer_returns_exact_binding_and_forwards_modes(
    tmp_path, monkeypatch, sealed_fresh, meter_fresh
):
    contract = _contract()
    devices = {"input": 8, "output": 9}
    followup = meter.build_fullband_v5_followup(
        contract,
        resolved_devices=devices,
        confirmations=_confirmations(),
    )
    raw = tmp_path / "meter_raw.npz"
    receipt = tmp_path / "meter_raw.receipt.json"
    raw.write_bytes(b"immutable meter raw fixture")
    receipt.write_bytes(b"immutable meter receipt fixture")
    completed = dt.datetime(2026, 8, 29, 1, 2, 3, tzinfo=dt.timezone.utc)
    metadata = {
        "fullband_v5_followup": followup,
        "fullband_v5_post_capture_revalidation": {"passed": True, "error": None},
        "repository_execution": _execution(),
    }
    observed = {}

    def static(**kwargs):
        observed["sealed"] = kwargs["require_sealed_raw_fresh"]
        return contract

    def generic(raw_path, **kwargs):
        observed["meter_fresh"] = kwargs["require_fresh"]
        assert raw_path == raw
        return {
            "path": raw,
            "receipt_path": receipt,
            "sha256": _file_sha(raw),
            "metadata": metadata,
            "completed_at_utc": completed,
            "meter_ch0_dbfs": -50.1,
        }

    monkeypatch.setattr(meter, "load_fullband_v5_static_contract", static)
    monkeypatch.setattr(meter, "resolve_fullband_v5_devices", lambda *_a, **_k: devices)
    monkeypatch.setattr(meter, "validate_bootstrap_meter_raw", generic)

    result = meter.validate_fullband_v5_meter_raw(
        raw,
        repository_root=tmp_path,
        require_fresh=meter_fresh,
        require_sealed_raw_fresh=sealed_fresh,
        sd_module=object(),
    )

    assert observed == {"sealed": sealed_fresh, "meter_fresh": meter_fresh}
    assert result["raw_sha256"] == _file_sha(raw)
    assert result["receipt_sha256"] == _file_sha(receipt)
    assert result["completed_at_utc"] == completed.isoformat()
    assert len(result["identity_sha256"]) == 64
    assert result["followup_contract_sha256"] == followup["followup_contract_sha256"]
    assert result["plan"] == contract["plan"]
    assert result["live_capture_authority"] == contract["live_capture_authority"]
    assert result["level_evidence"] == contract["level_evidence"]
    assert result["hardware"] == {
        **contract["hardware"],
        "resolved_devices": devices,
    }


def test_saved_meter_rejects_failed_post_revalidation(tmp_path, monkeypatch):
    contract = _contract()
    devices = {"input": 1, "output": 2}
    raw = tmp_path / "meter_raw.npz"
    receipt = tmp_path / "meter_raw.receipt.json"
    raw.write_bytes(b"raw")
    receipt.write_bytes(b"receipt")
    metadata = {
        "fullband_v5_followup": meter.build_fullband_v5_followup(
            contract,
            resolved_devices=devices,
            confirmations=_confirmations(),
        ),
        "fullband_v5_post_capture_revalidation": {
            "passed": False,
            "error": "injected drift",
        },
        "repository_execution": _execution(),
    }
    monkeypatch.setattr(meter, "load_fullband_v5_static_contract", lambda **_k: contract)
    monkeypatch.setattr(meter, "resolve_fullband_v5_devices", lambda *_a, **_k: devices)
    monkeypatch.setattr(
        meter,
        "validate_bootstrap_meter_raw",
        lambda *_a, **_k: {
            "path": raw,
            "receipt_path": receipt,
                "sha256": _file_sha(raw),
            "metadata": metadata,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc),
            "meter_ch0_dbfs": -50.1,
        },
    )

    with pytest.raises(ValueError, match="post-capture"):
        meter.validate_fullband_v5_meter_raw(
            raw,
            repository_root=tmp_path,
            sd_module=object(),
        )


def test_static_meter_validator_never_resolves_or_imports_backend(
    tmp_path, monkeypatch
):
    contract = _contract()
    devices = {"input": 4, "output": 5}
    raw = tmp_path / "meter_raw.npz"
    receipt = tmp_path / "meter_raw.receipt.json"
    raw.write_bytes(b"static meter raw")
    receipt.write_bytes(b"static meter receipt")
    completed = dt.datetime(2026, 8, 29, 1, 2, 3, tzinfo=dt.timezone.utc)
    metadata = {
        "fullband_v5_followup": meter.build_fullband_v5_followup(
            contract,
            resolved_devices=devices,
            confirmations=_confirmations(),
        ),
        "fullband_v5_post_capture_revalidation": {"passed": True, "error": None},
        "repository_execution": _execution(),
    }
    monkeypatch.setattr(
        meter, "load_fullband_v5_static_contract", lambda **_kwargs: contract
    )
    monkeypatch.setattr(
        meter,
        "validate_bootstrap_meter_raw",
        lambda *_args, **_kwargs: {
            "path": raw,
            "receipt_path": receipt,
            "sha256": _file_sha(raw),
            "metadata": metadata,
            "completed_at_utc": completed,
            "meter_ch0_dbfs": -50.1,
        },
    )
    monkeypatch.setattr(
        meter,
        "resolve_fullband_v5_devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("static meter validation must not resolve PortAudio")
        ),
    )

    result = meter.validate_fullband_v5_meter_raw_static(
        raw,
        repository_root=tmp_path,
    )

    assert result["hardware"]["resolved_devices"] == devices


def test_static_wrapper_forwards_exact_paths_without_backend(monkeypatch, tmp_path):
    observed = {}

    def load(**kwargs):
        observed.update(kwargs)
        return {"static": "PASS"}

    monkeypatch.setattr(meter, "load_fullband_v5_static_contract", load)
    result = meter.validate_fullband_v5_static_contract(
        repository_root=tmp_path,
        plan_envelope_path=meter.DEFAULT_PLAN_ENVELOPE_PATH,
        live_authority_path=meter.DEFAULT_LIVE_AUTHORITY_PATH,
        level_evidence_path=meter.DEFAULT_LEVEL_EVIDENCE_PATH,
        hardware_path=meter.DEFAULT_HARDWARE_PATH,
        raw_target_path=meter.DEFAULT_RAW_TARGET_PATH,
        require_sealed_raw_fresh=True,
    )

    assert result == {"static": "PASS"}
    assert observed == {
        "repository_root": tmp_path,
        "plan_envelope": meter.DEFAULT_PLAN_ENVELOPE_PATH,
        "live_authority": meter.DEFAULT_LIVE_AUTHORITY_PATH,
        "hardware": meter.DEFAULT_HARDWARE_PATH,
        "level_evidence": meter.DEFAULT_LEVEL_EVIDENCE_PATH,
        "raw_target": meter.DEFAULT_RAW_TARGET_PATH,
        "require_sealed_raw_fresh": True,
    }


def test_dirfd_publish_parent_rename_symlink_cannot_escape(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    parent = root / "results" / "meter"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    saved = root / "results" / "meter-held"
    real_link = repository_fd.os.link

    def attack_link(src, dst, **kwargs):
        parent.rename(saved)
        parent.symlink_to(outside, target_is_directory=True)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(repository_fd.os, "link", attack_link)
    with pytest.raises(RuntimeError, match="directory inode"):
        publish_repository_bytes_noreplace(
            root, "results/meter/raw.npz", b"captured"
        )
    assert not (outside / "raw.npz").exists()
    assert (saved / "raw.npz").read_bytes() == b"captured"


def test_dirfd_publish_final_target_race_is_noreplace(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    real_link = repository_fd.os.link

    def race_link(src, dst, **kwargs):
        fd = os.open(
            dst,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        os.write(fd, b"attacker")
        os.close(fd)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(repository_fd.os, "link", race_link)
    with pytest.raises(FileExistsError):
        publish_repository_bytes_noreplace(root, "results/raw.npz", b"captured")
    assert (root / "results/raw.npz").read_bytes() == b"attacker"
    assert not list((root / "results").glob("*.partial"))


def test_dirfd_publish_rejects_symlink_alias_and_existing_target(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside.bin"
    (root / "results").mkdir(parents=True)
    outside.write_bytes(b"outside")
    (root / "results/raw.npz").symlink_to(outside)
    with pytest.raises(FileExistsError):
        publish_repository_bytes_noreplace(root, "results/raw.npz", b"captured")
    assert outside.read_bytes() == b"outside"
    (root / "results/raw.npz").unlink()
    publish_repository_bytes_noreplace(root, "results/raw.npz", b"first")
    with pytest.raises(FileExistsError):
        publish_repository_bytes_noreplace(root, "results/raw.npz", b"second")
    assert (root / "results/raw.npz").read_bytes() == b"first"


def test_v5_writer_preserves_raw_when_receipt_publish_fails(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    real_publish = meter.publish_repository_bytes_noreplace
    calls = 0

    def fail_receipt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("receipt blocked")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(meter, "publish_repository_bytes_noreplace", fail_receipt)
    with pytest.raises(RuntimeError, match="receipt blocked"):
        meter.write_fullband_v5_meter_raw_atomic(
            "results/raw.npz",
            repository_root=tmp_path,
            metadata={"schema": "test"},
            submitted_output_pcm_int16=np.zeros((4, 2), dtype=np.int16),
            input_raw_int32=np.zeros((4, 2), dtype=np.int32),
        )
    assert (tmp_path / "results/raw.npz").is_file()
    assert not (tmp_path / "results/raw.receipt.json").exists()


def test_v5_writer_detects_raw_swap_before_receipt(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    real_publish = meter.publish_repository_bytes_noreplace

    def publish_then_swap(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        target = tmp_path / result["path"]
        target.unlink()
        target.write_bytes(b"forged")
        return result

    monkeypatch.setattr(meter, "publish_repository_bytes_noreplace", publish_then_swap)
    with pytest.raises(RuntimeError, match="publish 결과"):
        meter.write_fullband_v5_meter_raw_atomic(
            "results/raw.npz",
            repository_root=tmp_path,
            metadata={"schema": "test"},
            submitted_output_pcm_int16=np.zeros((4, 2), dtype=np.int16),
            input_raw_int32=np.zeros((4, 2), dtype=np.int32),
        )
    assert not (tmp_path / "results/raw.receipt.json").exists()


@pytest.mark.parametrize("alias", ["raw.npz", "raw.receipt.json"])
def test_v5_writer_rejects_raw_or_receipt_symlink_alias(tmp_path, alias):
    (tmp_path / "results").mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (tmp_path / "results" / alias).symlink_to(outside)
    expected = FileExistsError if alias == "raw.npz" else RuntimeError
    with pytest.raises(expected):
        meter.write_fullband_v5_meter_raw_atomic(
            "results/raw.npz",
            repository_root=tmp_path,
            metadata={"schema": "test"},
            submitted_output_pcm_int16=np.zeros((4, 2), dtype=np.int16),
            input_raw_int32=np.zeros((4, 2), dtype=np.int32),
        )
    assert outside.read_bytes() == b"outside"
    if alias == "raw.receipt.json":
        assert (tmp_path / "results/raw.npz").is_file()


def _write_small_v5_meter(root: Path):
    return meter.write_fullband_v5_meter_raw_atomic(
        "results/raw.npz",
        repository_root=root,
        metadata={"schema": "test"},
        submitted_output_pcm_int16=np.zeros((4, 2), dtype=np.int16),
        input_raw_int32=np.zeros((4, 2), dtype=np.int32),
    )


def test_v5_writer_success_retains_private_recovery_evidence(tmp_path):
    (tmp_path / "results").mkdir()
    result = _write_small_v5_meter(tmp_path)
    assert result["raw"].is_file()
    assert result["receipt"].is_file()
    recoveries = list((tmp_path / "results").glob(".*.v5_raw_recovery"))
    assert recoveries == [result["recovery"]]
    assert _file_sha(recoveries[0]) == result["sha256"]
    assert not list((tmp_path / "results").glob("*.partial"))


def test_v5_writer_raw_unlink_during_receipt_preserves_original_recovery(
    monkeypatch, tmp_path
):
    (tmp_path / "results").mkdir()
    real_publish = meter.publish_repository_bytes_noreplace
    calls = 0
    original_sha = None

    def attack(*args, **kwargs):
        nonlocal calls, original_sha
        calls += 1
        if calls == 2:
            raw = tmp_path / "results/raw.npz"
            original_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            raw.unlink()
            raw.write_bytes(b"forged")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(meter, "publish_repository_bytes_noreplace", attack)
    with pytest.raises(RuntimeError, match="원본 recovery="):
        _write_small_v5_meter(tmp_path)
    recoveries = list((tmp_path / "results").glob(".*.v5_raw_recovery"))
    assert len(recoveries) == 1
    assert hashlib.sha256(recoveries[0].read_bytes()).hexdigest() == original_sha
    assert (tmp_path / "results/raw.npz").read_bytes() == b"forged"


@pytest.mark.parametrize("receipt_verify_count", [2, 3])
def test_v5_writer_receipt_swap_after_snapshot_or_prior_verify_is_detected(
    monkeypatch, tmp_path, receipt_verify_count
):
    (tmp_path / "results").mkdir()
    real_verify = RepositoryFileGuard.verify
    count = 0

    def attack_verify(self):
        nonlocal count
        real_verify(self)
        if self.label == "v5 meter receipt":
            count += 1
            if count == receipt_verify_count:
                receipt = tmp_path / "results/raw.receipt.json"
                receipt.unlink()
                receipt.write_text("forged", encoding="utf-8")

    monkeypatch.setattr(RepositoryFileGuard, "verify", attack_verify)
    with pytest.raises(RuntimeError, match="원본 recovery="):
        _write_small_v5_meter(tmp_path)
    recoveries = list((tmp_path / "results").glob(".*.v5_raw_recovery"))
    assert len(recoveries) == 1


def test_execution_identity_rejects_assume_unchanged_and_head_blob_mismatch(tmp_path):
    root = tmp_path / "repo"
    script = root / "scripts/run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('committed')\n", encoding="utf-8")
    dependency = root / "src/dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    identity = repository_execution_identity(root, "scripts/run.py")
    assert identity["script_file_sha256"] == _file_sha(script)

    subprocess.run(
        ["git", "-C", str(root), "update-index", "--assume-unchanged", "src/dependency.py"],
        check=True,
    )
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="assume-unchanged"):
        repository_execution_identity(root, "scripts/run.py")
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--no-assume-unchanged", "src/dependency.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "checkout", "--", "src/dependency.py"], check=True
    )
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--skip-worktree", "src/dependency.py"],
        check=True,
    )
    with pytest.raises(RuntimeError, match="skip-worktree"):
        repository_execution_identity(root, "scripts/run.py")
