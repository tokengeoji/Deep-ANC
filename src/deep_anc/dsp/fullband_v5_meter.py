"""Committed causal v5 레벨 미터와 후속 capture 결속의 공용 계약.

이 모듈은 스피커를 출력하지 않는다. tracked signal plan/live capture authority,
hardware, paired level evidence와 fresh sealed raw target을 검증하고, 20초 meter raw에
삽입할 exact followup payload를 만든다. ``set_amp_level.py``와 live adapter는 이
모듈만 공유하며 서로 script를 import하지 않는다.
"""

from __future__ import annotations

import datetime as dt
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml
import numpy as np

from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    assert_repository_target_fresh_nofollow,
    canonical_relative_path,
    external_post_receipt_relative_path_v5,
    publish_repository_bytes_noreplace,
    repository_root,
    repository_execution_identity,
)
from deep_anc.dsp import fullband_live_authority_v5 as authority
from deep_anc.dsp.measurement_level import (
    BOOTSTRAP_METER_RECEIPT_SCHEMA,
    BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS,
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH,
    MEASUREMENT_LEVEL_EVIDENCE_SCHEMA,
    OFFICIAL_MEASUREMENT_CHANNEL_MAP,
    OFFICIAL_MEASUREMENT_LEVEL,
    measurement_hardware_identity,
    meter_receipt_path,
    require_physical_hardware_identity,
    validate_bootstrap_meter_raw,
    validate_measurement_hardware_contract,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _lexical_repository_relative(
    value: str | os.PathLike[str], *, repository_root_path: Path
) -> str:
    supplied_input = Path(os.fspath(value))
    supplied = (
        supplied_input
        if supplied_input.is_absolute()
        else repository_root_path / supplied_input
    )
    supplied = Path(os.path.abspath(supplied))
    try:
        relative = supplied.relative_to(repository_root_path)
    except ValueError as error:
        raise ValueError("meter raw target은 repository 내부여야 합니다") from error
    return canonical_relative_path(relative.as_posix(), label="meter raw target")


def write_fullband_v5_meter_raw_atomic(
    raw_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    metadata: dict[str, Any],
    submitted_output_pcm_int16: np.ndarray,
    input_raw_int32: np.ndarray,
    _generation_label: str = "v5",
    _recovery_tag: str = "v5_raw",
) -> dict[str, Any]:
    """Publish meter NPZ and receipt through guarded dirfd operations.

    두 private 인자는 v6 thin wrapper만 사용하며 기존 v5 호출의 bytes/path는 동일하다.
    """

    if (_generation_label, _recovery_tag) not in {
        ("v5", "v5_raw"),
        ("v6", "v6_raw"),
    }:
        raise ValueError("meter raw generation/recovery tag 조합이 유효하지 않습니다")

    root = Path(os.path.abspath(os.fspath(repository_root)))
    raw_relative = _lexical_repository_relative(
        raw_path, repository_root_path=root
    )
    receipt_relative = meter_receipt_path(raw_relative).as_posix()
    safe_metadata = json.loads(_canonical_json(metadata))
    safe_metadata["durable_raw_recovery"] = {
        "retained_on_success_and_failure": True,
        "same_inode_hardlink_not_duplicate_capture": True,
        "path_and_sha_reported_by_writer": True,
        "suffix": f".{_recovery_tag}_recovery",
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray(_canonical_json(safe_metadata)),
        submitted_output_pcm_int16=np.asarray(
            submitted_output_pcm_int16, dtype=np.int16
        ),
        input_raw_int32=np.asarray(input_raw_int32, dtype=np.int32),
    )
    raw_bytes = stream.getvalue()
    raw_result = publish_repository_bytes_noreplace(
        root,
        raw_relative,
        raw_bytes,
        preserve_recovery_link=True,
        recovery_tag=_recovery_tag,
    )
    recovery_relative = raw_result.get("recovery_path")
    if not isinstance(recovery_relative, str):
        raise RuntimeError(f"{_generation_label} meter raw recovery hardlink가 생성되지 않았습니다")
    receipt_payload = _canonical_json(
        {
            "schema": BOOTSTRAP_METER_RECEIPT_SCHEMA,
            "raw_path": raw_relative,
            "raw_sha256": raw_result["sha256"],
        }
    ).encode("utf-8")
    # Pin the raw inode while publishing and verifying its receipt.  A swap
    # before guard acquisition fails the expected digest/inode comparison.
    try:
        with RepositoryFileGuard(
            root, recovery_relative, label=f"{_generation_label} meter raw recovery"
        ) as recovery_guard:
            recovery_snapshot = recovery_guard.snapshot()
            if (
                recovery_snapshot["sha256"] != raw_result["sha256"]
                or recovery_snapshot["inode"] != raw_result["inode"]
                or recovery_snapshot["bytes"] != raw_bytes
            ):
                raise RuntimeError(f"{_generation_label} meter raw recovery inode 검증에 실패했습니다")
            with RepositoryFileGuard(
                root, raw_relative, label=f"{_generation_label} meter raw"
            ) as raw_guard:
                raw_snapshot = raw_guard.snapshot()
                if (
                    raw_snapshot["sha256"] != raw_result["sha256"]
                    or raw_snapshot["size"] != raw_result["size"]
                    or raw_snapshot["inode"] != raw_result["inode"]
                    or raw_snapshot["bytes"] != raw_bytes
                ):
                    raise RuntimeError(
                        f"{_generation_label} meter raw publish 결과가 pinned inode와 다릅니다"
                    )
                receipt_result = publish_repository_bytes_noreplace(
                    root, receipt_relative, receipt_payload
                )
                with RepositoryFileGuard(
                    root, receipt_relative, label=f"{_generation_label} meter receipt"
                ) as receipt_guard:
                    receipt_snapshot = receipt_guard.snapshot()
                    if (
                        receipt_snapshot["bytes"] != receipt_payload
                        or receipt_snapshot["sha256"] != receipt_result["sha256"]
                        or receipt_snapshot["inode"] != receipt_result["inode"]
                    ):
                        raise RuntimeError(
                            f"{_generation_label} meter receipt 최종 검증에 실패했습니다"
                        )
                    raw_guard.verify()
                    receipt_guard.verify()
                    raw_guard.verify()
                    recovery_guard.verify()
                    # Recovery is deliberately retained on success.  This final
                    # receipt verification is followed by no filesystem mutation.
                    receipt_guard.verify()
    except BaseException as error:
        raise RuntimeError(
            f"{_generation_label} meter raw/receipt durable publication 실패; 원본 recovery="
            f"{recovery_relative}: {error}"
        ) from error
    return {
        "raw": root.joinpath(*PurePosixPath(raw_relative).parts),
        "receipt": root.joinpath(*PurePosixPath(receipt_relative).parts),
        "sha256": raw_result["sha256"],
        "recovery": root.joinpath(*PurePosixPath(recovery_relative).parts),
        "recovery_relative_path": recovery_relative,
        "recovery_sha256": raw_result["sha256"],
    }


FULLBAND_V5_FOLLOWUP_SCHEMA = "fullband_causal_v5_meter_followup_v1"
FULLBAND_V5_FOLLOWUP_SCOPE = (
    "fresh_level_meter_binding_only_not_capture_plant_or_training_authority"
)
FULLBAND_V5_METER_IDENTITY_SCHEMA = "fullband_causal_v5_meter_identity_v1"
TRACKED_V5_LEVEL_ATTESTATION_SCOPE = (
    "tracked_historical_attestation_for_fresh_v5_meter_only"
)
EXPECTED_TRACKED_LEVEL_EVIDENCE_FILE_SHA256 = (
    "c76ac0d3c52c20fadd761d1ed0c85e27e3599328f60ca0d164535594336e73d0"
)

DEFAULT_PLAN_ENVELOPE_PATH = authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH
DEFAULT_LIVE_AUTHORITY_PATH = authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
DEFAULT_HARDWARE_PATH = authority.SEALED_HARDWARE_RELATIVE_PATH
DEFAULT_LEVEL_EVIDENCE_PATH = str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH)
DEFAULT_RAW_TARGET_PATH = authority.SEALED_RAW_RELATIVE_PATH

FOLLOWUP_KEYS = {
    "schema",
    "scope",
    "signal_plan",
    "live_capture_authority",
    "hardware",
    "level_evidence",
    "sealed_raw",
    "resolved_devices",
    "operator_confirmations",
    "followup_contract_sha256",
}
PLAN_KEYS = {"path", "file_sha256", "payload_sha256", "pcm_sha256"}
AUTHORITY_KEYS = {"path", "file_sha256", "payload_sha256"}
HARDWARE_KEYS = {
    "path",
    "file_sha256",
    "identity_sha256",
    "physical_fingerprint_sha256",
}
EVIDENCE_KEYS = {
    "path",
    "file_sha256",
    "identity_sha256",
    "scope",
    "preserved_raw_revalidated",
}
RAW_KEYS = {"path", "must_not_exist_before_capture"}
DEVICE_KEYS = {"input", "output"}
CONFIRMATION_KEYS = {
    "speaker_output",
    "user_present",
    "volume_minimum",
    "routing_and_geometry",
    "same_amplifier_setting",
}

_TRACKED_EVIDENCE_KEYS = {
    "capture_gap_seconds",
    "created_at_utc",
    "hardware_identity",
    "interleaved_err_noise_bin_dbfs",
    "interleaved_err_noise_bin_target_dbfs",
    "interleaved_err_noise_bin_tolerance_db",
    "interleaved_raw",
    "max_capture_gap_seconds",
    "meter_ch0_dbfs",
    "meter_raw",
    "meter_target_dbfs",
    "meter_tolerance_db",
    "passed",
    "probe_amplitude",
    "same_amplifier_setting",
    "sample_rate",
    "schema",
}
_PINNED_METER_RAW_REFERENCE = {
    "completed_at_utc": "2026-08-27T02:25:34.061512+00:00",
    "path": (
        "results/calibration_interleaved/level_bootstrap/"
        "20260827_112512_5dc06fdd/meter_raw.npz"
    ),
    "sha256": "c0169ef42ab29ed738bf197ce62e21d6f47404207cba1a41ab88e4bb8cd87f31",
    "status": "PASS",
}
_PINNED_INTERLEAVED_RAW_REFERENCE = {
    "path": (
        "results/calibration_interleaved/strict_20260827/"
        "20260827_112608_5ac13134/raw_measurement.npz"
    ),
    "sha256": "31d563b163fe7dcb3f6b85e30e491a6775947e7f1b988690c3668fd13464b347",
    "started_at_utc": "2026-08-27T02:26:12.492322+00:00",
    "status": "PASS",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} key 집합이 exact하지 않습니다")
    return dict(value)


def _repository_root(value: str | os.PathLike[str]) -> Path:
    return repository_root(value)


def _repository_relative(path: Path, *, repository_root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = lexical.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact가 저장소 밖을 가리킵니다: {path}") from exc
    return canonical_relative_path(relative, label="artifact path")


def _exact_repository_path(
    supplied: str | Path,
    expected_relative: str,
    *,
    repository_root: Path,
    label: str,
    require_file: bool,
) -> Path:
    expected_relative = canonical_relative_path(expected_relative, label=label)
    expected = Path(os.path.abspath(os.fspath(repository_root / expected_relative)))
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if candidate != expected:
        raise ValueError(f"{label} path는 exact {expected_relative}여야 합니다")
    # 파일/parent 검사는 caller가 dirfd/O_NOFOLLOW guard를 연 상태에서 수행한다.
    # 여기서 Path.exists()/is_file()을 사용하면 check→open 경합이 다시 생긴다.
    del require_file
    return expected


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"tracked v5 level evidence duplicate key: {key}")
        result[key] = item
    return result


def _load_canonical_compact_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}가 UTF-8 JSON이 아닙니다") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} top-level은 object여야 합니다")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} bytes가 canonical compact JSON이 아닙니다")
    return value


def _parse_exact_utc(value: Any, *, label: str) -> dt.datetime:
    if type(value) is not str:
        raise ValueError(f"{label}는 UTC ISO-8601 문자열이어야 합니다")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label}가 유효한 ISO-8601이 아닙니다") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{label}는 UTC offset을 가져야 합니다")
    return parsed.astimezone(dt.timezone.utc)


def _validate_tracked_v5_level_attestation_bytes(
    raw: bytes,
    *,
    relative_path: str,
) -> dict[str, Any]:
    """Tracked historical bytes만 검증한다; preserved raw PASS를 주장하지 않는다."""

    if relative_path != str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH):
        raise ValueError("tracked v5 level attestation path가 pinned evidence와 다릅니다")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != EXPECTED_TRACKED_LEVEL_EVIDENCE_FILE_SHA256:
        raise ValueError("tracked v5 level evidence file SHA가 pinned 값과 다릅니다")
    payload = _load_canonical_compact_json(raw, label="tracked v5 level evidence")
    if set(payload) != _TRACKED_EVIDENCE_KEYS:
        raise ValueError("tracked v5 level evidence key 집합이 exact하지 않습니다")
    if payload.get("schema") != MEASUREMENT_LEVEL_EVIDENCE_SCHEMA:
        raise ValueError("tracked v5 level evidence schema가 다릅니다")
    if payload.get("passed") is not True or payload.get("same_amplifier_setting") is not True:
        raise ValueError("tracked v5 level evidence PASS/amplifier 확인이 없습니다")

    exact_values: dict[str, int | float] = {
        "sample_rate": OFFICIAL_MEASUREMENT_LEVEL.sample_rate,
        "probe_amplitude": OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        "meter_target_dbfs": OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
        "meter_tolerance_db": OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
        "interleaved_err_noise_bin_target_dbfs": (
            OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_dbfs
        ),
        "interleaved_err_noise_bin_tolerance_db": (
            OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_tolerance_db
        ),
        "max_capture_gap_seconds": BOOTSTRAP_METER_MAX_AGE_SECONDS,
    }
    for name, expected in exact_values.items():
        observed = payload.get(name)
        if type(expected) is int:
            matches = type(observed) is int and observed == expected
        else:
            matches = type(observed) in {int, float} and float(observed) == float(expected)
        if not matches:
            raise ValueError(
                f"tracked v5 level evidence {name}가 official 값과 다릅니다"
            )
    if payload.get("meter_raw") != _PINNED_METER_RAW_REFERENCE:
        raise ValueError("tracked v5 meter raw reference가 pinned 값과 다릅니다")
    if payload.get("interleaved_raw") != _PINNED_INTERLEAVED_RAW_REFERENCE:
        raise ValueError("tracked v5 interleaved raw reference가 pinned 값과 다릅니다")

    meter_completed = _parse_exact_utc(
        _PINNED_METER_RAW_REFERENCE["completed_at_utc"],
        label="tracked meter completion",
    )
    strict_started = _parse_exact_utc(
        _PINNED_INTERLEAVED_RAW_REFERENCE["started_at_utc"],
        label="tracked interleaved start",
    )
    created = _parse_exact_utc(payload.get("created_at_utc"), label="tracked evidence creation")
    gap = (strict_started - meter_completed).total_seconds()
    observed_gap = payload.get("capture_gap_seconds")
    if type(observed_gap) not in {int, float} or abs(float(observed_gap) - gap) > 1e-6:
        raise ValueError("tracked v5 level evidence capture gap가 raw refs와 다릅니다")
    if (
        gap < -BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS
        or gap > BOOTSTRAP_METER_MAX_AGE_SECONDS
        or created < strict_started
    ):
        raise ValueError("tracked v5 level evidence 시간 계약이 유효하지 않습니다")

    try:
        meter_dbfs = float(payload["meter_ch0_dbfs"])
        strict_dbfs = float(payload["interleaved_err_noise_bin_dbfs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tracked v5 level evidence observed dBFS가 없습니다") from error
    if abs(meter_dbfs - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs) > (
        OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db
    ):
        raise ValueError("tracked v5 meter dBFS가 official tolerance 밖입니다")
    if abs(strict_dbfs - OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_dbfs) > (
        OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_tolerance_db
    ):
        raise ValueError("tracked v5 interleaved dBFS가 official tolerance 밖입니다")

    identity = payload.get("hardware_identity")
    if not isinstance(identity, dict):
        raise ValueError("tracked v5 level evidence hardware identity가 없습니다")
    require_physical_hardware_identity(identity)
    if (
        identity.get("sample_rate") != OFFICIAL_MEASUREMENT_LEVEL.sample_rate
        or identity.get("block_size") != 256
        or identity.get("latency") != "low"
        or identity.get("channel_map") != OFFICIAL_MEASUREMENT_CHANNEL_MAP
    ):
        raise ValueError("tracked v5 level evidence logical hardware가 official 계약과 다릅니다")
    for direction in ("input", "output"):
        endpoint = identity.get(direction)
        if not isinstance(endpoint, dict) or set(endpoint) != {
            "card",
            "channels",
            "pcm",
        }:
            raise ValueError(f"tracked v5 level evidence {direction} identity가 exact하지 않습니다")
        if type(endpoint.get("channels")) is not int or endpoint["channels"] != 2:
            raise ValueError(f"tracked v5 level evidence {direction} channels가 2가 아닙니다")

    return {
        "path": relative_path,
        "file_sha256": actual_sha,
        "identity_sha256": payload_sha256(identity),
        "scope": TRACKED_V5_LEVEL_ATTESTATION_SCOPE,
        "preserved_raw_revalidated": False,
        "strict_ps_authority": False,
        "plant_or_training_authority": False,
        "live_admission_eligible": False,
        "hardware_identity": json.loads(canonical_json_bytes(identity).decode("utf-8")),
        "evidence": payload,
    }


def load_tracked_v5_level_attestation(
    evidence_path: str | Path,
    *,
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Clean exact checkout에서 raw 없이 읽을 수 있는 좁은 v5 attestation."""

    root = _repository_root(repository_root)
    expected = _exact_repository_path(
        evidence_path,
        str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
        repository_root=root,
        label="tracked v5 level evidence",
        require_file=True,
    )
    relative = expected.relative_to(root).as_posix()
    with RepositoryFileGuard(root, relative, label="tracked v5 level evidence") as guard:
        result = _validate_tracked_v5_level_attestation_bytes(
            guard.bytes, relative_path=relative
        )
        guard.verify()
        return result


def _load_saved_authority_without_raw_freshness_gate(
    *, repository_root: Path, authority_raw: bytes
) -> dict[str, Any]:
    """publish 뒤 raw 존재만 허용하고 authority bytes/의미는 exact 재검증한다."""

    raw = bytes(authority_raw)
    if hashlib.sha256(raw).hexdigest() != authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256:
        raise ValueError("v5 live authority file SHA가 pinned 값과 다릅니다")

    def reject_duplicates(pairs):  # noqa: ANN001
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"v5 live authority duplicate key: {key}")
            result[key] = item
        return result

    try:
        saved = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v5 live authority가 UTF-8 JSON이 아닙니다") from exc
    canonical_file = (
        json.dumps(
            saved,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if canonical_file != raw:
        raise ValueError("v5 live authority file bytes가 canonical JSON이 아닙니다")
    expected = authority.build_live_capture_authority_v5(
        repository_root=repository_root,
        expected_plan_envelope_file_sha256=(
            authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256
        ),
        expected_condition_receipt_payload_sha256=(
            authority.EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
        expected_hardware_file_sha256=authority.EXPECTED_HARDWARE_FILE_SHA256,
    )
    if saved != expected:
        raise ValueError("v5 live authority payload가 pinned builder exact 결과와 다릅니다")
    if saved.get("authority_sha256") != authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256:
        raise ValueError("v5 live authority payload SHA가 pinned 값과 다릅니다")
    return {
        "path": authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        "file_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        "payload_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        "authority": saved,
    }


def load_fullband_v5_static_contract(
    *,
    repository_root: str | os.PathLike[str],
    plan_envelope: str | Path = DEFAULT_PLAN_ENVELOPE_PATH,
    live_authority: str | Path = DEFAULT_LIVE_AUTHORITY_PATH,
    hardware: str | Path = DEFAULT_HARDWARE_PATH,
    level_evidence: str | Path = DEFAULT_LEVEL_EVIDENCE_PATH,
    raw_target: str | Path = DEFAULT_RAW_TARGET_PATH,
    require_sealed_raw_fresh: bool,
    physical_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """PortAudio import 전에 tracked attestation과 hardware 계약을 검증한다.

    과거 paired raw를 요구하는 strict forensic loader를 약화하지 않는다. 이 live-v5
    경로는 exact tracked evidence bytes에서 좁은 historical attestation만 읽고,
    fresh 20초 v5 meter가 뒤이어 현재 레벨을 다시 증명하도록 한다.

    ``physical_fingerprint``를 주면 호출자가 이미 수집한 현재 ALSA fingerprint와
    tracked attestation을 비교한다. 생략하면 이 함수는 의도적으로 호스트 ALSA를
    열지 않고, attestation에 봉인된 fingerprint를 portable static 값으로 사용한다.
    따라서 Elice 같은 오디오 장치가 없는 학습 노드에서도 파일·계약 검증을 수행할
    수 있으며, 실제 live capture 경로는 반드시 현재 fingerprint를 명시적으로
    전달해야 한다.
    """

    root = _repository_root(repository_root)
    plan_path = _exact_repository_path(
        plan_envelope,
        authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        repository_root=root,
        label="v5 signal plan",
        require_file=True,
    )
    authority_path = _exact_repository_path(
        live_authority,
        authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        repository_root=root,
        label="v5 live authority",
        require_file=True,
    )
    hardware_path = _exact_repository_path(
        hardware,
        authority.SEALED_HARDWARE_RELATIVE_PATH,
        repository_root=root,
        label="v5 hardware",
        require_file=True,
    )
    evidence_path = _exact_repository_path(
        level_evidence,
        str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
        repository_root=root,
        label="v5 level evidence",
        require_file=True,
    )
    _exact_repository_path(
        raw_target,
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=root,
        label="v5 sealed raw",
        require_file=False,
    )
    raw_relative = authority.SEALED_RAW_RELATIVE_PATH
    with ExitStack() as stack:
        guards = [
            stack.enter_context(
                RepositoryFileGuard(
                    root,
                    authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
                    label="v5 signal plan",
                )
            ),
            stack.enter_context(
                RepositoryFileGuard(
                    root,
                    authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
                    label="v5 live authority",
                )
            ),
            stack.enter_context(
                RepositoryFileGuard(
                    root,
                    authority.SEALED_HARDWARE_RELATIVE_PATH,
                    label="v5 hardware",
                )
            ),
            stack.enter_context(
                RepositoryFileGuard(
                    root,
                    str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
                    label="tracked v5 level evidence",
                )
            ),
        ]
        plan_guard, authority_guard, hardware_guard, evidence_guard = guards
        raw_guard: RepositoryFileGuard | None = None
        if require_sealed_raw_fresh:
            assert_repository_target_fresh_nofollow(
                root, raw_relative, create_parents=False
            )
            post_receipt_relative = external_post_receipt_relative_path_v5(
                raw_relative
            )
            assert_repository_target_fresh_nofollow(
                root, post_receipt_relative, create_parents=False
            )
        else:
            raw_guard = stack.enter_context(
                RepositoryFileGuard(root, raw_relative, label="v5 sealed raw")
            )

        loaded_plan = authority.load_exact_saved_plan_v5(
            plan_path,
            repository_root=root,
            expected_file_sha256=authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            expected_condition_receipt_payload_sha256=(
                authority.EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
            ),
        )
        if require_sealed_raw_fresh:
            loaded_authority = authority.load_exact_saved_live_capture_authority_v5(
                authority_path,
                repository_root=root,
                expected_file_sha256=(
                    authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
                ),
                expected_payload_sha256=(
                    authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256
                ),
            )
        else:
            loaded_authority = _load_saved_authority_without_raw_freshness_gate(
                repository_root=root,
                authority_raw=authority_guard.bytes,
            )

        if hardware_guard.sha256 != authority.EXPECTED_HARDWARE_FILE_SHA256:
            raise ValueError("v5 hardware file SHA가 pinned authority와 다릅니다")
        try:
            hardware_config = yaml.safe_load(
                hardware_guard.bytes.decode("utf-8", errors="strict")
            )
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError("v5 hardware YAML bytes를 해석할 수 없습니다") from error
        if not isinstance(hardware_config, dict):
            raise ValueError("v5 hardware YAML top-level은 mapping이어야 합니다")
        hardware_audio, channel_map = validate_measurement_hardware_contract(
            hardware_config
        )
        attestation = _validate_tracked_v5_level_attestation_bytes(
            evidence_guard.bytes,
            relative_path=str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
        )
        if physical_fingerprint is None:
            # static-only/portable 검증은 호스트의 /proc/asound·/sysfs를 읽지 않는다.
            # tracked attestation 자체가 physical identity의 canonical snapshot이다.
            physical_fingerprint = attestation["hardware_identity"].get(
                "physical_fingerprint"
            )
        if not isinstance(physical_fingerprint, Mapping):
            raise ValueError("v5 physical fingerprint mapping이 필요합니다")
        physical_fingerprint = json.loads(_canonical_json(physical_fingerprint))
        hardware_identity = measurement_hardware_identity(
            hardware_config,
            physical_fingerprint=physical_fingerprint,
        )
        if attestation["hardware_identity"] != hardware_identity:
            raise ValueError(
                "tracked v5 level attestation과 현재 hardware/physical identity가 다릅니다"
            )

        # Path-based legacy semantic builders가 호출된 구간 전체에서 같은 file fds와
        # 모든 parent dirfds를 유지했으며, 현재 bytes/inode/name을 다시 확인한다.
        for guard in guards:
            guard.verify()
        if raw_guard is not None:
            raw_guard.verify()
        else:
            assert_repository_target_fresh_nofollow(
                root, raw_relative, create_parents=False
            )
            assert_repository_target_fresh_nofollow(
                root, post_receipt_relative, create_parents=False
            )

    identity_sha = payload_sha256(hardware_identity)
    physical_sha = str(physical_fingerprint.get("sha256", ""))
    if len(physical_sha) != 64:
        raise ValueError("v5 physical fingerprint SHA가 필요합니다")
    return {
        "plan": {
            "path": authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": loaded_plan["file_sha256"],
            "payload_sha256": loaded_plan["payload_sha256"],
            "pcm_sha256": loaded_plan["pcm_sha256"],
        },
        "live_capture_authority": {
            "path": authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
            "file_sha256": loaded_authority["file_sha256"],
            "payload_sha256": loaded_authority["payload_sha256"],
        },
        "hardware": {
            "path": authority.SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": authority.EXPECTED_HARDWARE_FILE_SHA256,
            "identity_sha256": identity_sha,
            "physical_fingerprint_sha256": physical_sha,
        },
        "level_evidence": {
            "path": str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
            "file_sha256": attestation["file_sha256"],
            "identity_sha256": identity_sha,
            "scope": TRACKED_V5_LEVEL_ATTESTATION_SCOPE,
            "preserved_raw_revalidated": False,
        },
        "sealed_raw": {
            "path": authority.SEALED_RAW_RELATIVE_PATH,
            "must_not_exist_before_capture": True,
        },
        "hardware_config": hardware_config,
        "hardware_audio": hardware_audio,
        "channel_map": channel_map,
        "hardware_identity": hardware_identity,
        "physical_fingerprint": physical_fingerprint,
        "evidence": attestation["evidence"],
        "portable_level_attestation": attestation,
    }


def validate_fullband_v5_static_contract(
    *,
    repository_root: str | os.PathLike[str],
    plan_envelope_path: str | Path = DEFAULT_PLAN_ENVELOPE_PATH,
    live_authority_path: str | Path = DEFAULT_LIVE_AUTHORITY_PATH,
    level_evidence_path: str | Path = DEFAULT_LEVEL_EVIDENCE_PATH,
    hardware_path: str | Path = DEFAULT_HARDWARE_PATH,
    raw_target_path: str | Path = DEFAULT_RAW_TARGET_PATH,
    require_sealed_raw_fresh: bool,
    physical_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """live adapter용 명시적 static-only public wrapper (PortAudio 접근 없음)."""

    kwargs: dict[str, Any] = {
        "repository_root": repository_root,
        "plan_envelope": plan_envelope_path,
        "live_authority": live_authority_path,
        "hardware": hardware_path,
        "level_evidence": level_evidence_path,
        "raw_target": raw_target_path,
        "require_sealed_raw_fresh": require_sealed_raw_fresh,
    }
    # None은 의도적으로 전달하지 않아 기존 static wrapper의 exact forwarding과
    # portable 호출 의미를 유지한다.
    if physical_fingerprint is not None:
        kwargs["physical_fingerprint"] = physical_fingerprint
    return load_fullband_v5_static_contract(**kwargs)


def resolve_fullband_v5_devices(
    contract: Mapping[str, Any], *, sd_module=None
) -> dict[str, int]:  # noqa: ANN001
    from deep_anc.audio_io import resolve_alsa_portaudio_device

    if sd_module is None:
        import sounddevice as sd_module

    # resolve_alsa_portaudio_device imports the same backend internally; sd_module is an
    # explicit admission marker/test seam and prevents callers from resolving before the
    # static contract. Query identity remains in the shared audio primitive.
    del sd_module
    hardware = contract["hardware_audio"]
    devices = {
        "input": resolve_alsa_portaudio_device(
            hardware["input"]["card"], hardware["input"]["pcm"], "input", 2
        ),
        "output": resolve_alsa_portaudio_device(
            hardware["output"]["card"], hardware["output"]["pcm"], "output", 2
        ),
    }
    if any(type(value) is not int or value < 0 for value in devices.values()):
        raise ValueError("v5 resolved device는 음이 아닌 exact int여야 합니다")
    return devices


def build_fullband_v5_followup(
    contract: Mapping[str, Any],
    *,
    resolved_devices: Mapping[str, int],
    confirmations: Mapping[str, bool],
) -> dict[str, Any]:
    devices = _exact_mapping(resolved_devices, DEVICE_KEYS, label="resolved devices")
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("resolved devices는 음이 아닌 exact int여야 합니다")
    confirmed = _exact_mapping(
        confirmations, CONFIRMATION_KEYS, label="v5 operator confirmations"
    )
    if any(confirmed[name] is not True for name in confirmed):
        raise ValueError("v5 meter의 다섯 operator confirmation이 모두 필요합니다")
    core = {
        "schema": FULLBAND_V5_FOLLOWUP_SCHEMA,
        "scope": FULLBAND_V5_FOLLOWUP_SCOPE,
        "signal_plan": dict(contract["plan"]),
        "live_capture_authority": dict(contract["live_capture_authority"]),
        "hardware": dict(contract["hardware"]),
        "level_evidence": dict(contract["level_evidence"]),
        "sealed_raw": dict(contract["sealed_raw"]),
        "resolved_devices": devices,
        "operator_confirmations": confirmed,
    }
    return {**core, "followup_contract_sha256": payload_sha256(core)}


def validate_fullband_v5_followup(
    value: Any,
    *,
    expected_contract: Mapping[str, Any],
    expected_devices: Mapping[str, int],
) -> dict[str, Any]:
    followup = _exact_mapping(value, FOLLOWUP_KEYS, label="fullband v5 meter followup")
    if followup["schema"] != FULLBAND_V5_FOLLOWUP_SCHEMA:
        raise ValueError("old broadband/v4 meter followup schema를 거부합니다")
    if followup["scope"] != FULLBAND_V5_FOLLOWUP_SCOPE:
        raise ValueError("fullband v5 meter followup scope가 다릅니다")
    _exact_mapping(followup["signal_plan"], PLAN_KEYS, label="v5 followup plan")
    _exact_mapping(
        followup["live_capture_authority"],
        AUTHORITY_KEYS,
        label="v5 followup authority",
    )
    _exact_mapping(followup["hardware"], HARDWARE_KEYS, label="v5 followup hardware")
    _exact_mapping(followup["level_evidence"], EVIDENCE_KEYS, label="v5 followup evidence")
    _exact_mapping(followup["sealed_raw"], RAW_KEYS, label="v5 followup raw")
    devices = _exact_mapping(
        followup["resolved_devices"], DEVICE_KEYS, label="v5 followup devices"
    )
    confirmations = _exact_mapping(
        followup["operator_confirmations"],
        CONFIRMATION_KEYS,
        label="v5 followup confirmations",
    )
    if any(confirmations[name] is not True for name in confirmations):
        raise ValueError("v5 followup 다섯 확인이 모두 exact true여야 합니다")
    core = {key: item for key, item in followup.items() if key != "followup_contract_sha256"}
    if followup["followup_contract_sha256"] != payload_sha256(core):
        raise ValueError("v5 meter followup contract SHA가 payload와 다릅니다")
    expected = build_fullband_v5_followup(
        expected_contract,
        resolved_devices=expected_devices,
        confirmations={name: True for name in CONFIRMATION_KEYS},
    )
    if followup != expected or devices != dict(expected_devices):
        raise ValueError("v5 meter followup가 현재 plan/authority/hardware/evidence/device와 다릅니다")
    return json.loads(canonical_json_bytes(followup).decode("utf-8"))


def _validate_fullband_v5_meter_raw_static_impl(
    raw_path: str | Path,
    *,
    repository_root: str | os.PathLike[str],
    now_utc: dt.datetime | None = None,
    require_fresh: bool = True,
    require_sealed_raw_fresh: bool = True,
) -> dict[str, Any]:
    """PortAudio 없이 saved PASS meter bytes/followup을 검증한다."""

    root = _repository_root(repository_root)
    raw_relative = _repository_relative(Path(raw_path), repository_root=root)
    anticipated_receipt = meter_receipt_path(
        root.joinpath(*PurePosixPath(raw_relative).parts)
    )
    receipt_relative = _repository_relative(anticipated_receipt, repository_root=root)
    with ExitStack() as stack:
        raw_guard = stack.enter_context(
            RepositoryFileGuard(root, raw_relative, label="fullband v5 meter raw")
        )
        receipt_guard = stack.enter_context(
            RepositoryFileGuard(
                root, receipt_relative, label="fullband v5 meter receipt"
            )
        )
        contract = load_fullband_v5_static_contract(
            repository_root=root,
            require_sealed_raw_fresh=bool(require_sealed_raw_fresh),
        )
        verified = validate_bootstrap_meter_raw(
            root.joinpath(*PurePosixPath(raw_relative).parts),
            repository_root=root,
            expected_hardware_identity=contract["hardware_identity"],
            now_utc=now_utc,
            require_fresh=bool(require_fresh),
        )
        raw = Path(verified["path"])
        receipt = Path(verified["receipt_path"])
        if (
            _repository_relative(raw, repository_root=root) != raw_relative
            or _repository_relative(receipt, repository_root=root) != receipt_relative
        ):
            raise ValueError("v5 meter validator가 다른 raw/receipt path를 반환했습니다")
        raw_sha = raw_guard.sha256
        receipt_sha = receipt_guard.sha256
        if raw_sha != verified["sha256"]:
            raise ValueError("v5 meter raw가 held-fd bytes와 다릅니다")
        raw_followup = verified["metadata"].get("fullband_v5_followup")
        if not isinstance(raw_followup, Mapping):
            raise ValueError("v5 meter raw에 fullband followup mapping이 없습니다")
        devices = _exact_mapping(
            raw_followup.get("resolved_devices"),
            DEVICE_KEYS,
            label="v5 embedded resolved devices",
        )
        if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
            raise ValueError("v5 embedded resolved devices가 음이 아닌 exact int가 아닙니다")
        followup = validate_fullband_v5_followup(
            raw_followup,
            expected_contract=contract,
            expected_devices=devices,
        )
        saved_execution = verified["metadata"].get("repository_execution")
        current_execution = repository_execution_identity(
            root, "scripts/data/set_amp_level.py"
        )
        if saved_execution != current_execution:
            raise ValueError(
                "v5 meter repository commit/branch/script SHA가 current clean checkout과 다릅니다"
            )
        post = _exact_mapping(
            verified["metadata"].get("fullband_v5_post_capture_revalidation"),
            {"passed", "error"},
            label="fullband v5 meter post-capture revalidation",
        )
        if post != {"passed": True, "error": None}:
            raise ValueError("v5 meter post-capture binding이 PASS가 아닙니다")
        raw_guard.verify()
        receipt_guard.verify()

    completed = verified["completed_at_utc"].isoformat()
    identity_payload = {
        "schema": FULLBAND_V5_METER_IDENTITY_SCHEMA,
        "path": _repository_relative(raw, repository_root=root),
        "receipt_path": _repository_relative(receipt, repository_root=root),
        "raw_sha256": raw_sha,
        "receipt_sha256": receipt_sha,
        "completed_at_utc": completed,
        "followup_contract_sha256": followup["followup_contract_sha256"],
    }
    return {
        "path": raw,
        "receipt_path": receipt,
        "raw_sha256": raw_sha,
        "receipt_sha256": receipt_sha,
        "metadata": verified["metadata"],
        "completed_at_utc": completed,
        "meter_ch0_dbfs": float(verified["meter_ch0_dbfs"]),
        "identity_sha256": payload_sha256(identity_payload),
        "followup_contract_sha256": followup["followup_contract_sha256"],
        "plan": dict(contract["plan"]),
        "live_capture_authority": dict(contract["live_capture_authority"]),
        "level_evidence": dict(contract["level_evidence"]),
        "hardware": {
            **dict(contract["hardware"]),
            "resolved_devices": dict(devices),
        },
        "sealed_raw": dict(contract["sealed_raw"]),
        "_static_contract": contract,
    }


def validate_fullband_v5_meter_raw_static(
    raw_path: str | Path,
    *,
    repository_root: str | os.PathLike[str],
    now_utc: dt.datetime | None = None,
    require_fresh: bool = True,
    require_sealed_raw_fresh: bool = True,
) -> dict[str, Any]:
    """sounddevice import 전에 meter raw/receipt/followup을 완전히 검증한다."""

    result = _validate_fullband_v5_meter_raw_static_impl(
        raw_path,
        repository_root=repository_root,
        now_utc=now_utc,
        require_fresh=require_fresh,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )
    return {key: item for key, item in result.items() if key != "_static_contract"}


def validate_fullband_v5_meter_raw(
    raw_path: str | Path,
    *,
    repository_root: str | os.PathLike[str],
    now_utc: dt.datetime | None = None,
    require_fresh: bool = True,
    require_sealed_raw_fresh: bool = True,
    sd_module=None,
) -> dict[str, Any]:  # noqa: ANN001
    """static PASS meter와 현재 PortAudio device index를 함께 검증한다."""

    result = _validate_fullband_v5_meter_raw_static_impl(
        raw_path,
        repository_root=repository_root,
        now_utc=now_utc,
        require_fresh=require_fresh,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )
    contract = result.pop("_static_contract")
    if sd_module is None:
        import sounddevice as sd_module
    current_devices = resolve_fullband_v5_devices(contract, sd_module=sd_module)
    if result["hardware"]["resolved_devices"] != current_devices:
        raise ValueError("v5 meter embedded devices가 current PortAudio devices와 다릅니다")
    return result


__all__ = [
    "CONFIRMATION_KEYS",
    "DEFAULT_HARDWARE_PATH",
    "DEFAULT_LEVEL_EVIDENCE_PATH",
    "DEFAULT_LIVE_AUTHORITY_PATH",
    "DEFAULT_PLAN_ENVELOPE_PATH",
    "DEFAULT_RAW_TARGET_PATH",
    "EXPECTED_TRACKED_LEVEL_EVIDENCE_FILE_SHA256",
    "FULLBAND_V5_FOLLOWUP_SCHEMA",
    "FULLBAND_V5_FOLLOWUP_SCOPE",
    "FULLBAND_V5_METER_IDENTITY_SCHEMA",
    "TRACKED_V5_LEVEL_ATTESTATION_SCOPE",
    "build_fullband_v5_followup",
    "load_fullband_v5_static_contract",
    "load_tracked_v5_level_attestation",
    "resolve_fullband_v5_devices",
    "validate_fullband_v5_followup",
    "validate_fullband_v5_meter_raw",
    "validate_fullband_v5_meter_raw_static",
    "validate_fullband_v5_static_contract",
    "write_fullband_v5_meter_raw_atomic",
]
