from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from deep_anc.dsp.fullband_live_authority_v5 import (
    AUTHORITY_SCHEMA,
    EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256,
    EXPECTED_HARDWARE_FILE_SHA256,
    EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
    EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
    EXPECTED_PCM_SHA256,
    EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
    EXPECTED_PLAN_PAYLOAD_SHA256,
    SEALED_HARDWARE_RELATIVE_PATH,
    SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
    SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
    SEALED_RAW_RELATIVE_PATH,
    build_live_capture_authority_v5,
    canonical_plan_envelope_bytes_v5,
    load_exact_saved_live_capture_authority_v5,
    load_exact_saved_plan_v5,
    validate_live_capture_authority_v5,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    plan_path = root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    hardware_path = root / SEALED_HARDWARE_RELATIVE_PATH
    raw_parent = (root / SEALED_RAW_RELATIVE_PATH).parent
    plan_path.parent.mkdir(parents=True)
    hardware_path.parent.mkdir(parents=True)
    raw_parent.mkdir(parents=True)
    plan_bytes = canonical_plan_envelope_bytes_v5()
    hardware_bytes = (PROJECT_ROOT / SEALED_HARDWARE_RELATIVE_PATH).read_bytes()
    plan_path.write_bytes(plan_bytes)
    hardware_path.write_bytes(hardware_bytes)
    assert _sha(plan_bytes) == EXPECTED_PLAN_ENVELOPE_FILE_SHA256
    assert _sha(hardware_bytes) == EXPECTED_HARDWARE_FILE_SHA256
    return root, EXPECTED_PLAN_ENVELOPE_FILE_SHA256, EXPECTED_HARDWARE_FILE_SHA256


def _authority(root: Path, plan_sha: str, hardware_sha: str) -> dict:
    return build_live_capture_authority_v5(
        repository_root=root,
        expected_plan_envelope_file_sha256=plan_sha,
        expected_condition_receipt_payload_sha256=(
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
        expected_hardware_file_sha256=hardware_sha,
    )


def _write_authority_asset(root: Path, authority: dict) -> Path:
    path = root / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(authority))
    return path


def _reseal(authority: dict) -> dict:
    result = copy.deepcopy(authority)
    core = {key: value for key, value in result.items() if key != "authority_sha256"}
    result["authority_sha256"] = _payload_sha(core)
    return result


def test_exact_saved_plan_and_capture_only_authority_pass(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    loaded = load_exact_saved_plan_v5(
        root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        repository_root=root,
        expected_file_sha256=plan_sha,
        expected_condition_receipt_payload_sha256=(
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
    )
    assert loaded["file_sha256"] == plan_sha
    assert loaded["payload_sha256"] == EXPECTED_PLAN_PAYLOAD_SHA256
    assert loaded["pcm_sha256"] == EXPECTED_PCM_SHA256
    assert loaded["sealed_raw_relative_path"] == SEALED_RAW_RELATIVE_PATH
    condition = loaded["envelope"]["support_1024_condition_receipt"]
    assert (
        condition["canonical_payload_sha256"]
        == EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    )

    authority = _authority(root, plan_sha, hardware_sha)
    receipt = validate_live_capture_authority_v5(
        authority,
        repository_root=root,
        expected_authority_sha256=authority["authority_sha256"],
    )
    assert receipt["schema"] == AUTHORITY_SCHEMA
    assert receipt["capture_only"] is True
    assert receipt["canonical_training_eligible"] is False
    assert receipt["hardware_sample_slip_authority"] is False
    assert receipt["live_delay_authority"] is None
    assert receipt["sealed_raw"] == {
        "path": SEALED_RAW_RELATIVE_PATH,
        "fresh": True,
    }


def test_actual_tracked_plan_and_hardware_match_pinned_bytes(tmp_path: Path) -> None:
    plan_path = PROJECT_ROOT / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    authority_path = PROJECT_ROOT / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    hardware_path = PROJECT_ROOT / SEALED_HARDWARE_RELATIVE_PATH
    assert _sha(plan_path.read_bytes()) == EXPECTED_PLAN_ENVELOPE_FILE_SHA256
    assert _sha(authority_path.read_bytes()) == (
        EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
    )
    assert _sha(hardware_path.read_bytes()) == EXPECTED_HARDWARE_FILE_SHA256
    loaded = load_exact_saved_plan_v5(
        plan_path,
        repository_root=PROJECT_ROOT,
        expected_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
        expected_condition_receipt_payload_sha256=(
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
    )
    authority = build_live_capture_authority_v5(
        repository_root=PROJECT_ROOT,
        expected_plan_envelope_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
        expected_condition_receipt_payload_sha256=(
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
        expected_hardware_file_sha256=EXPECTED_HARDWARE_FILE_SHA256,
    )
    assert loaded["path"] == SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    assert authority["signal_plan_envelope"]["file_sha256"] == (
        EXPECTED_PLAN_ENVELOPE_FILE_SHA256
    )
    assert authority["signal_plan_envelope"][
        "condition_receipt_payload_sha256"
    ] == EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    assert authority["hardware"]["file_sha256"] == EXPECTED_HARDWARE_FILE_SHA256
    assert _canonical(authority) == authority_path.read_bytes()
    # 이 테스트의 목적은 tracked plan/authority/hardware bytes 검증이다. 실제 장비에
    # immutable v5 raw가 이미 존재해도 asset 검증이 환경 의존적으로 실패하면 안 된다.
    # canonical 세 파일만 fresh 임시 root에 복제해 pre-capture freshness까지 함께 본다.
    isolated_plan = tmp_path / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    isolated_authority = tmp_path / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    isolated_hardware = tmp_path / SEALED_HARDWARE_RELATIVE_PATH
    isolated_plan.parent.mkdir(parents=True)
    isolated_authority.parent.mkdir(parents=True, exist_ok=True)
    isolated_hardware.parent.mkdir(parents=True, exist_ok=True)
    isolated_plan.write_bytes(plan_path.read_bytes())
    isolated_authority.write_bytes(authority_path.read_bytes())
    isolated_hardware.write_bytes(hardware_path.read_bytes())
    loaded_authority = load_exact_saved_live_capture_authority_v5(
        isolated_authority,
        repository_root=tmp_path,
        expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
    )
    assert loaded_authority["file_sha256"] == (
        EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
    )
    assert loaded_authority["payload_sha256"] == (
        EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256
    )


def test_live_preflight_never_runs_heavy_exact_condition_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deep_anc.dsp import fullband_causal_v5 as signal_v5

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("live preflight에서 heavy eigensolve를 호출했습니다")

    monkeypatch.setattr(signal_v5, "exact_condition_audit_v5", forbidden)
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    loaded = load_exact_saved_plan_v5(
        root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        repository_root=root,
        expected_file_sha256=plan_sha,
        expected_condition_receipt_payload_sha256=(
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
    )
    authority = _authority(root, plan_sha, hardware_sha)
    receipt = validate_live_capture_authority_v5(
        authority,
        repository_root=root,
        expected_authority_sha256=authority["authority_sha256"],
    )
    assert loaded["file_sha256"] == EXPECTED_PLAN_ENVELOPE_FILE_SHA256
    assert receipt["capture_only"] is True


def test_plan_requires_external_file_sha_and_canonical_file_bytes(tmp_path: Path) -> None:
    root, plan_sha, _ = _make_repository(tmp_path)
    plan_path = root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    with pytest.raises(ValueError, match="file SHA"):
        load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256="0" * 64,
            expected_condition_receipt_payload_sha256=(
                EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
            ),
        )
    with pytest.raises(ValueError, match="condition receipt payload SHA"):
        load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            expected_condition_receipt_payload_sha256="0" * 64,
        )

    parsed = json.loads(plan_path.read_text(encoding="utf-8"))
    noncanonical = json.dumps(parsed, sort_keys=True).encode("utf-8")
    plan_path.write_bytes(noncanonical)
    with pytest.raises(ValueError, match="file SHA"):
        load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            expected_condition_receipt_payload_sha256=(
                EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
            ),
        )
    assert plan_sha != _sha(noncanonical)


@pytest.mark.parametrize("mutation", ["extra_key", "swapped_plan", "legacy_schema"])
def test_plan_envelope_rejects_tamper_and_old_schema(
    tmp_path: Path, mutation: str
) -> None:
    root, _, _ = _make_repository(tmp_path)
    plan_path = root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    envelope = json.loads(plan_path.read_text(encoding="utf-8"))
    if mutation == "extra_key":
        envelope["unexpected"] = True
    elif mutation == "swapped_plan":
        envelope["signal_plan"]["role"] = "swapped_diagnostic_plan"
    else:
        envelope["schema"] = "broadband_interleaved_measurement_plan_v4"
    raw = _canonical(envelope)
    plan_path.write_bytes(raw)
    with pytest.raises(ValueError, match="file SHA"):
        load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            expected_condition_receipt_payload_sha256=(
                EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
            ),
        )


def test_authority_requires_external_sha_and_rejects_tamper(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    with pytest.raises(ValueError, match="external anchor"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256="0" * 64,
        )

    changed = copy.deepcopy(authority)
    changed["capture_only"] = False
    with pytest.raises(ValueError, match="SHA"):
        validate_live_capture_authority_v5(
            changed,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_exact_saved_authority_loader_passes_canonical_temp_asset(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    authority_path = _write_authority_asset(root, authority)
    assert _sha(authority_path.read_bytes()) == (
        EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
    )
    loaded = load_exact_saved_live_capture_authority_v5(
        authority_path,
        repository_root=root,
        expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
    )
    assert loaded["authority"] == authority
    assert loaded["validation"]["capture_only"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("reformat", "canonical JSON"),
        ("duplicate", "duplicate key"),
        ("tamper", "file SHA"),
        ("old_schema", "legacy/v4/old broadband"),
    ],
)
def test_saved_authority_rejects_reformat_tamper_duplicate_and_old_schema(
    tmp_path: Path, mutation: str, match: str
) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    authority_path = root / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    if mutation == "reformat":
        raw = json.dumps(authority, sort_keys=True).encode("utf-8")
    elif mutation == "duplicate":
        raw = _canonical(authority).replace(
            b"{\n",
            b'{\n  "schema": "fullband_causal_v4_live_capture_authority",\n',
            1,
        )
    elif mutation == "tamper":
        authority["capture_only"] = False
        authority = _reseal(authority)
        raw = _canonical(authority)
    else:
        authority["schema"] = "broadband_interleaved_live_authority_v4"
        authority = _reseal(authority)
        raw = _canonical(authority)
    authority_path.write_bytes(raw)
    with pytest.raises(ValueError, match=match):
        load_exact_saved_live_capture_authority_v5(
            authority_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        )


def test_saved_authority_requires_pinned_external_file_and_payload_sha(
    tmp_path: Path,
) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority_path = _write_authority_asset(
        root, _authority(root, plan_sha, hardware_sha)
    )
    with pytest.raises(ValueError, match="file SHA"):
        load_exact_saved_live_capture_authority_v5(
            authority_path,
            repository_root=root,
            expected_file_sha256="0" * 64,
            expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        )
    with pytest.raises(ValueError, match="payload SHA"):
        load_exact_saved_live_capture_authority_v5(
            authority_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            expected_payload_sha256="0" * 64,
        )


def test_saved_authority_rejects_nonsealed_repository_path(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    wrong_path = root / "assets/contracts/copied_live_authority.json"
    wrong_path.write_bytes(_canonical(authority))
    with pytest.raises(ValueError, match="sealed repository path"):
        load_exact_saved_live_capture_authority_v5(
            wrong_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        )


def test_saved_authority_rejects_symlink_and_preexisting_raw(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path / "symlink")
    authority = _authority(root, plan_sha, hardware_sha)
    authority_path = root / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    outside = tmp_path / "outside-authority.json"
    outside.write_bytes(_canonical(authority))
    authority_path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_exact_saved_live_capture_authority_v5(
            authority_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        )

    root, plan_sha, hardware_sha = _make_repository(tmp_path / "existing")
    authority_path = _write_authority_asset(
        root, _authority(root, plan_sha, hardware_sha)
    )
    (root / SEALED_RAW_RELATIVE_PATH).write_bytes(b"immutable-existing-raw")
    with pytest.raises(FileExistsError, match="이미 존재"):
        load_exact_saved_live_capture_authority_v5(
            authority_path,
            repository_root=root,
            expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (
            "plan_path",
            "results/data_audit/broadband_measurement_signal_plan_v4.json",
            "plan path",
        ),
        ("hardware_path", "configs/old_hardware.yaml", "hardware path"),
        ("raw_path", "results/fullband_causal_v4/raw_capture.npz", "raw path"),
        ("plan_sha", "1" * 64, "plan envelope file SHA"),
        ("condition_sha", "3" * 64, "condition receipt SHA"),
        ("hardware_sha", "2" * 64, "hardware SHA"),
    ],
)
def test_resealed_authority_still_rejects_wrong_path_or_file_sha(
    tmp_path: Path, field: str, replacement: str, match: str
) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    if field == "plan_path":
        authority["signal_plan_envelope"]["path"] = replacement
    elif field == "hardware_path":
        authority["hardware"]["path"] = replacement
    elif field == "raw_path":
        authority["sealed_raw"]["path"] = replacement
    elif field == "plan_sha":
        authority["signal_plan_envelope"]["file_sha256"] = replacement
    elif field == "condition_sha":
        authority["signal_plan_envelope"][
            "condition_receipt_payload_sha256"
        ] = replacement
    else:
        authority["hardware"]["file_sha256"] = replacement
    authority = _reseal(authority)
    with pytest.raises((ValueError, FileExistsError), match=match):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_plan_hardware_and_raw_parent_symlinks_are_rejected(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path / "plan")
    plan_path = root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    outside_plan = tmp_path / "outside-plan.json"
    outside_plan.write_bytes(plan_path.read_bytes())
    plan_path.unlink()
    plan_path.symlink_to(outside_plan)
    with pytest.raises(ValueError, match="symlink"):
        load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256=plan_sha,
            expected_condition_receipt_payload_sha256=(
                EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
            ),
        )

    root, plan_sha, hardware_sha = _make_repository(tmp_path / "hardware")
    authority = _authority(root, plan_sha, hardware_sha)
    hardware_path = root / SEALED_HARDWARE_RELATIVE_PATH
    outside_hardware = tmp_path / "outside-hardware.yaml"
    outside_hardware.write_bytes(hardware_path.read_bytes())
    hardware_path.unlink()
    hardware_path.symlink_to(outside_hardware)
    with pytest.raises(ValueError, match="symlink"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )

    root, plan_sha, hardware_sha = _make_repository(tmp_path / "raw")
    authority = _authority(root, plan_sha, hardware_sha)
    raw_parent = (root / SEALED_RAW_RELATIVE_PATH).parent
    outside_raw = tmp_path / "outside-raw"
    outside_raw.mkdir()
    raw_parent.rmdir()
    raw_parent.symlink_to(outside_raw, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_preexisting_or_broken_symlink_raw_target_is_not_fresh(tmp_path: Path) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path / "existing")
    authority = _authority(root, plan_sha, hardware_sha)
    raw_path = root / SEALED_RAW_RELATIVE_PATH
    raw_path.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="이미 존재"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )

    root, plan_sha, hardware_sha = _make_repository(tmp_path / "broken")
    authority = _authority(root, plan_sha, hardware_sha)
    raw_path = root / SEALED_RAW_RELATIVE_PATH
    raw_path.symlink_to(tmp_path / "missing-target.npz")
    with pytest.raises(ValueError, match="symlink"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )


@pytest.mark.parametrize(
    "legacy_schema",
    [
        "fullband_causal_v4_live_capture_authority",
        "broadband_interleaved_live_authority_v5",
    ],
)
def test_old_authority_schema_is_rejected_even_when_resealed(
    tmp_path: Path, legacy_schema: str
) -> None:
    root, plan_sha, hardware_sha = _make_repository(tmp_path)
    authority = _authority(root, plan_sha, hardware_sha)
    authority["schema"] = legacy_schema
    authority = _reseal(authority)
    with pytest.raises(ValueError, match="legacy/v4/old broadband"):
        validate_live_capture_authority_v5(
            authority,
            repository_root=root,
            expected_authority_sha256=authority["authority_sha256"],
        )
