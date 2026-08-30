"""v5 live capture를 열기 전 파일 결속만 검증하는 capture-only primitive.

이 모듈은 오디오 장치를 열거나 raw를 쓰지 않는다. 검토된 authority SHA를 외부에서
전달받아 committed v5 signal-plan envelope, hardware 설정 bytes, 아직 존재하지 않는
sealed raw target을 fail-closed로 대조한다. 이 증거만으로 plant/training authority는
절대 열리지 않는다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .fullband_causal_v5 import build_plan_v5


AUTHORITY_SCHEMA = "fullband_causal_v5_live_capture_authority_v1"
PLAN_ENVELOPE_SCHEMA = "fullband_causal_signal_plan_envelope_v5"
PLAN_SCHEMA = "fullband_causal_time_separated_near_white_v5"

SEALED_PLAN_ENVELOPE_RELATIVE_PATH = (
    "assets/contracts/fullband_causal_v5_signal_plan.json"
)
SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH = (
    "assets/contracts/fullband_causal_v5_live_capture_authority.json"
)
SEALED_RAW_RELATIVE_PATH = "results/fullband_causal_v5/raw_capture.npz"
SEALED_HARDWARE_RELATIVE_PATH = "configs/hardware_jetson.yaml"

EXPECTED_PLAN_PAYLOAD_SHA256 = (
    "32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127"
)
EXPECTED_PCM_SHA256 = (
    "c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff"
)
EXPECTED_PLAN_ENVELOPE_FILE_SHA256 = (
    "bf25f041c5c5770c01aa326e47749b4eaab9a012f9f7c69dec5cd81ae3507287"
)
EXPECTED_HARDWARE_FILE_SHA256 = (
    "45232a45e51fd76c7b88db338b9cf4f3840a88299b4d452e259064c0ee559351"
)
EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256 = (
    "300078f714fd19e6b15eaee1bc212b196960301a1c745c256d3d46ac9295b61f"
)
EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256 = (
    "f090e59533fac6467f3c7c3328ebc9983deef06d8f8bc6fbf9158e4de66f8138"
)
EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256 = (
    "ed506255f53056724abd2fd79822e91b8879455b9cdf0c06ab942c079ae9441f"
)

_PINNED_CONDITION_RECEIPT: dict[str, Any] = {
    "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
    "canonical_payload_sha256": EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256,
    "fit_roles": ["fit_a", "fit_b"],
    "joint_fit_condition_number": 9.058033530917806,
    "longer_supports": {
        "2048": "NOT_AUDITED_NO_CLAIM",
        "4096": "NOT_AUDITED_NO_CLAIM",
        "8192": "NOT_AUDITED_NO_CLAIM",
    },
    "maximum_allowed": 20.0,
    "maximum_eigenvalue": 1013588159.6722435,
    "minimum_eigenvalue": 111899360.5193181,
    "passed": True,
    "role_condition_numbers": {
        "fit_a": 11.571714021472085,
        "fit_b": 12.575291092522631,
    },
    "schema": "fullband_causal_exact_gram_condition_v5",
    "support_samples": 1024,
}

_ENVELOPE_KEYS = {
    "schema",
    "signal_plan",
    "support_1024_condition_receipt",
}
_PLAN_REFERENCE_KEYS = {
    "path",
    "file_sha256",
    "payload_sha256",
    "pcm_sha256",
    "condition_receipt_payload_sha256",
}
_FILE_REFERENCE_KEYS = {"path", "file_sha256"}
_RAW_REFERENCE_KEYS = {"path", "must_not_exist_before_capture"}
_AUTHORITY_KEYS = {
    "schema",
    "scope",
    "capture_only",
    "plan_live_capture_enabled",
    "canonical_training_eligible",
    "hardware_sample_slip_authority",
    "live_delay_authority",
    "signal_plan_envelope",
    "hardware",
    "sealed_raw",
    "authority_sha256",
}


def _canonical_json_file_bytes(value: object) -> bytes:
    """기존 v5 signal-plan writer와 동일한 canonical JSON file bytes."""

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


def _canonical_payload_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_payload_bytes(value)).hexdigest()


def _file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{label}는 64자리 소문자 SHA-256이어야 합니다")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label}는 64자리 소문자 SHA-256이어야 합니다")
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} key 집합이 exact하지 않습니다")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON duplicate key를 거부합니다: {key}")
        result[key] = value
    return result


def _load_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}가 유효한 UTF-8 JSON이 아닙니다") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} top-level은 object여야 합니다")
    try:
        canonical = _canonical_json_file_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}가 canonical JSON으로 직렬화되지 않습니다") from error
    if raw != canonical:
        raise ValueError(f"{label} bytes가 canonical JSON과 다릅니다")
    return value


def _canonical_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label}는 비어 있지 않은 repository 상대경로여야 합니다")
    if "\\" in value:
        raise ValueError(f"{label}에 backslash를 허용하지 않습니다")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise ValueError(f"{label}가 canonical repository 상대경로가 아닙니다")
    return value


def _repository_root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("repository root는 symlink가 아닌 기존 directory여야 합니다")
    return root


def _repository_path(
    repository_root: Path,
    relative_path: str,
    *,
    label: str,
    require_file: bool,
) -> Path:
    relative = _canonical_relative_path(relative_path, label=label)
    target = Path(os.path.abspath(repository_root.joinpath(*PurePosixPath(relative).parts)))
    try:
        target.relative_to(repository_root)
    except ValueError as error:  # pragma: no cover - canonical path의 이중 방어
        raise ValueError(f"{label}가 repository 밖을 가리킵니다") from error

    cursor = repository_root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} symlink를 거부합니다: {relative}")
        if index < len(parts) - 1 and cursor.exists() and not cursor.is_dir():
            raise ValueError(f"{label} parent가 directory가 아닙니다: {relative}")
    if require_file:
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"{label}가 symlink 아닌 기존 file이 아닙니다")
    return target


@lru_cache(maxsize=1)
def _expected_committed_plan() -> dict[str, Any]:
    """빠른 deterministic builder만 실행한다; heavy eigensolve는 호출하지 않는다."""

    plan, _ = build_plan_v5()
    if plan.get("canonical_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise AssertionError("committed v5 builder plan SHA가 pinned constant와 다릅니다")
    if plan.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256:
        raise AssertionError("committed v5 builder PCM SHA가 pinned constant와 다릅니다")
    return plan


def _validate_pinned_condition_receipt(value: object) -> dict[str, Any]:
    condition = _exact_keys(
        value,
        set(_PINNED_CONDITION_RECEIPT),
        label="support-1024 condition receipt",
    )
    if condition != _PINNED_CONDITION_RECEIPT:
        raise ValueError("support-1024 condition receipt가 pinned exact fields와 다릅니다")
    declared = _require_sha256(
        condition.get("canonical_payload_sha256"),
        label="condition receipt payload SHA",
    )
    core = {
        key: item
        for key, item in condition.items()
        if key != "canonical_payload_sha256"
    }
    if (
        declared != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        or _payload_sha256(core) != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        or condition.get("schema") != "fullband_causal_exact_gram_condition_v5"
        or condition.get("support_samples") != 1024
        or condition.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
        or condition.get("fit_roles") != ["fit_a", "fit_b"]
        or condition.get("passed") is not True
    ):
        raise ValueError("support-1024 condition receipt SHA/field/PASS gate가 실패했습니다")
    return dict(condition)


def committed_plan_envelope_v5() -> dict[str, Any]:
    """테스트/발행 상위 계층이 사용할 exact committed envelope 값을 반환한다."""

    plan = _expected_committed_plan()
    return {
        "schema": PLAN_ENVELOPE_SCHEMA,
        "signal_plan": copy.deepcopy(plan),
        "support_1024_condition_receipt": copy.deepcopy(_PINNED_CONDITION_RECEIPT),
    }


def canonical_plan_envelope_bytes_v5() -> bytes:
    """기존 signal-only writer와 byte-compatible한 exact envelope bytes."""

    payload = _canonical_json_file_bytes(committed_plan_envelope_v5())
    if _file_sha256_bytes(payload) != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise AssertionError("pinned v5 plan envelope canonical bytes SHA가 다릅니다")
    return payload


def load_exact_saved_plan_v5(
    plan_envelope_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    expected_file_sha256: str,
    expected_condition_receipt_payload_sha256: str,
) -> dict[str, Any]:
    """저장된 committed v5 plan envelope를 bytes부터 다시 검증한다."""

    expected_file = _require_sha256(
        expected_file_sha256, label="expected plan envelope file SHA"
    )
    if expected_file != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise ValueError("expected plan envelope file SHA가 pinned 값과 다릅니다")
    expected_condition = _require_sha256(
        expected_condition_receipt_payload_sha256,
        label="expected condition receipt payload SHA",
    )
    if expected_condition != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256:
        raise ValueError("expected condition receipt payload SHA가 pinned 값과 다릅니다")
    root = _repository_root(repository_root)
    expected_path = _repository_path(
        root,
        SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        label="signal plan envelope path",
        require_file=True,
    )
    supplied = Path(os.path.abspath(os.fspath(plan_envelope_path)))
    if supplied != expected_path:
        raise ValueError("signal plan envelope path가 sealed repository path와 다릅니다")
    raw = expected_path.read_bytes()
    actual_file = _file_sha256_bytes(raw)
    if actual_file != expected_file or actual_file != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise ValueError("signal plan envelope file SHA가 expected/pinned 값과 다릅니다")
    envelope = _load_canonical_json(raw, label="signal plan envelope")
    _exact_keys(envelope, _ENVELOPE_KEYS, label="signal plan envelope")
    if envelope.get("schema") != PLAN_ENVELOPE_SCHEMA:
        raise ValueError("legacy/v4/old broadband signal plan envelope를 거부합니다")

    expected_plan = _expected_committed_plan()
    plan = envelope.get("signal_plan")
    if not isinstance(plan, dict) or plan != expected_plan:
        raise ValueError("signal plan이 committed v5 builder exact 결과와 다릅니다")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("signal plan schema가 committed causal v5가 아닙니다")
    if plan.get("canonical_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise ValueError("signal plan payload SHA가 pinned committed 값과 다릅니다")
    if plan.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256:
        raise ValueError("signal plan PCM SHA가 pinned committed 값과 다릅니다")
    if (
        plan.get("live_capture_enabled") is not False
        or plan.get("live_authority") is not None
        or plan.get("canonical_training_eligible") is not False
    ):
        raise ValueError("signal plan의 signal-only 권한 경계가 바뀌었습니다")
    raw_path = plan.get("publisher_contract", {}).get("raw_session_relative_path")
    if raw_path != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("signal plan sealed raw relative path가 committed 값과 다릅니다")

    _validate_pinned_condition_receipt(
        envelope.get("support_1024_condition_receipt")
    )

    return {
        "path": SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        "file_sha256": actual_file,
        "payload_sha256": EXPECTED_PLAN_PAYLOAD_SHA256,
        "pcm_sha256": EXPECTED_PCM_SHA256,
        "condition_receipt_payload_sha256": (
            EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        ),
        "sealed_raw_relative_path": SEALED_RAW_RELATIVE_PATH,
        "envelope": envelope,
    }


def build_live_capture_authority_v5(
    *,
    repository_root: str | os.PathLike[str],
    expected_plan_envelope_file_sha256: str,
    expected_condition_receipt_payload_sha256: str,
    expected_hardware_file_sha256: str,
) -> dict[str, Any]:
    """외부에서 검토한 두 file SHA로 capture-only authority payload를 만든다."""

    root = _repository_root(repository_root)
    plan = load_exact_saved_plan_v5(
        root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        repository_root=root,
        expected_file_sha256=expected_plan_envelope_file_sha256,
        expected_condition_receipt_payload_sha256=(
            expected_condition_receipt_payload_sha256
        ),
    )
    hardware_expected = _require_sha256(
        expected_hardware_file_sha256, label="expected hardware file SHA"
    )
    if hardware_expected != EXPECTED_HARDWARE_FILE_SHA256:
        raise ValueError("expected hardware file SHA가 pinned 값과 다릅니다")
    hardware_path = _repository_path(
        root,
        SEALED_HARDWARE_RELATIVE_PATH,
        label="hardware path",
        require_file=True,
    )
    hardware_actual = _file_sha256_bytes(hardware_path.read_bytes())
    if (
        hardware_actual != hardware_expected
        or hardware_actual != EXPECTED_HARDWARE_FILE_SHA256
    ):
        raise ValueError("hardware file SHA가 expected/pinned 값과 다릅니다")

    core: dict[str, Any] = {
        "schema": AUTHORITY_SCHEMA,
        "scope": "capture_only_not_plant_or_training_authority",
        "capture_only": True,
        "plan_live_capture_enabled": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "live_delay_authority": None,
        "signal_plan_envelope": {
            "path": plan["path"],
            "file_sha256": plan["file_sha256"],
            "payload_sha256": plan["payload_sha256"],
            "pcm_sha256": plan["pcm_sha256"],
            "condition_receipt_payload_sha256": plan[
                "condition_receipt_payload_sha256"
            ],
        },
        "hardware": {
            "path": SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": hardware_actual,
        },
        "sealed_raw": {
            "path": plan["sealed_raw_relative_path"],
            "must_not_exist_before_capture": True,
        },
    }
    return {**core, "authority_sha256": _payload_sha256(core)}


def validate_live_capture_authority_v5(
    authority: Mapping[str, Any],
    *,
    repository_root: str | os.PathLike[str],
    expected_authority_sha256: str,
) -> dict[str, Any]:
    """외부 SHA에 고정된 authority와 현재 bytes/raw freshness를 다시 대조한다."""

    root = _repository_root(repository_root)
    value = _exact_keys(authority, _AUTHORITY_KEYS, label="live capture authority")
    if value.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("legacy/v4/old broadband live authority schema를 거부합니다")
    declared_sha = _require_sha256(
        value.get("authority_sha256"), label="authority internal SHA"
    )
    external_sha = _require_sha256(
        expected_authority_sha256, label="expected authority SHA"
    )
    core = {key: item for key, item in value.items() if key != "authority_sha256"}
    recomputed = _payload_sha256(core)
    if declared_sha != recomputed or external_sha != recomputed:
        raise ValueError("live capture authority SHA가 internal/external anchor와 다릅니다")
    if (
        value.get("scope") != "capture_only_not_plant_or_training_authority"
        or value.get("capture_only") is not True
        or value.get("plan_live_capture_enabled") is not False
        or value.get("canonical_training_eligible") is not False
        or value.get("hardware_sample_slip_authority") is not False
        or value.get("live_delay_authority") is not None
    ):
        raise ValueError("capture-only authority 경계가 변경됐습니다")

    plan_ref = _exact_keys(
        value.get("signal_plan_envelope"),
        _PLAN_REFERENCE_KEYS,
        label="authority signal plan reference",
    )
    if plan_ref.get("path") != SEALED_PLAN_ENVELOPE_RELATIVE_PATH:
        raise ValueError("authority signal plan path가 sealed path와 다릅니다")
    if plan_ref.get("file_sha256") != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise ValueError("authority plan envelope file SHA가 pinned 값과 다릅니다")
    if (
        plan_ref.get("condition_receipt_payload_sha256")
        != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    ):
        raise ValueError("authority condition receipt SHA가 pinned 값과 다릅니다")
    plan = load_exact_saved_plan_v5(
        root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        repository_root=root,
        expected_file_sha256=_require_sha256(
            plan_ref.get("file_sha256"), label="authority plan file SHA"
        ),
        expected_condition_receipt_payload_sha256=_require_sha256(
            plan_ref.get("condition_receipt_payload_sha256"),
            label="authority condition receipt payload SHA",
        ),
    )
    if (
        plan_ref.get("payload_sha256") != plan["payload_sha256"]
        or plan_ref.get("pcm_sha256") != plan["pcm_sha256"]
    ):
        raise ValueError("authority plan payload/PCM SHA가 committed plan과 다릅니다")

    hardware_ref = _exact_keys(
        value.get("hardware"), _FILE_REFERENCE_KEYS, label="authority hardware reference"
    )
    if hardware_ref.get("path") != SEALED_HARDWARE_RELATIVE_PATH:
        raise ValueError("authority hardware path가 sealed path와 다릅니다")
    hardware_path = _repository_path(
        root,
        SEALED_HARDWARE_RELATIVE_PATH,
        label="hardware path",
        require_file=True,
    )
    hardware_expected = _require_sha256(
        hardware_ref.get("file_sha256"), label="authority hardware file SHA"
    )
    if hardware_expected != EXPECTED_HARDWARE_FILE_SHA256:
        raise ValueError("authority hardware SHA가 pinned 값과 다릅니다")
    hardware_actual = _file_sha256_bytes(hardware_path.read_bytes())
    if (
        hardware_actual != hardware_expected
        or hardware_actual != EXPECTED_HARDWARE_FILE_SHA256
    ):
        raise ValueError("authority hardware SHA가 현재/pinned file bytes와 다릅니다")

    raw_ref = _exact_keys(
        value.get("sealed_raw"), _RAW_REFERENCE_KEYS, label="authority sealed raw reference"
    )
    if (
        raw_ref.get("path") != plan["sealed_raw_relative_path"]
        or raw_ref.get("path") != SEALED_RAW_RELATIVE_PATH
        or raw_ref.get("must_not_exist_before_capture") is not True
    ):
        raise ValueError("authority sealed raw path/freshness 계약이 다릅니다")
    raw_path = _repository_path(
        root,
        SEALED_RAW_RELATIVE_PATH,
        label="sealed raw path",
        require_file=False,
    )
    if os.path.lexists(raw_path):
        raise FileExistsError("sealed raw target이 이미 존재합니다; overwrite를 거부합니다")

    return {
        "schema": AUTHORITY_SCHEMA,
        "authority_sha256": recomputed,
        "capture_only": True,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "plan": {
            "path": plan["path"],
            "file_sha256": plan["file_sha256"],
            "payload_sha256": plan["payload_sha256"],
            "pcm_sha256": plan["pcm_sha256"],
            "condition_receipt_payload_sha256": plan[
                "condition_receipt_payload_sha256"
            ],
        },
        "hardware": {
            "path": SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": hardware_actual,
        },
        "sealed_raw": {
            "path": SEALED_RAW_RELATIVE_PATH,
            "fresh": True,
        },
        "live_delay_authority": None,
    }


def load_exact_saved_live_capture_authority_v5(
    authority_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    """Tracked capture-only authority file을 bytes부터 live 의미까지 검증한다."""

    external_file_sha = _require_sha256(
        expected_file_sha256,
        label="expected live capture authority file SHA",
    )
    external_payload_sha = _require_sha256(
        expected_payload_sha256,
        label="expected live capture authority payload SHA",
    )
    if external_file_sha != EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256:
        raise ValueError("expected live capture authority file SHA가 pinned 값과 다릅니다")
    if external_payload_sha != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256:
        raise ValueError("expected live capture authority payload SHA가 pinned 값과 다릅니다")

    root = _repository_root(repository_root)
    expected_path = _repository_path(
        root,
        SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        label="live capture authority path",
        require_file=False,
    )
    supplied = Path(os.path.abspath(os.fspath(authority_path)))
    if supplied != expected_path:
        raise ValueError("live capture authority path가 sealed repository path와 다릅니다")
    expected_path = _repository_path(
        root,
        SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        label="live capture authority path",
        require_file=True,
    )
    raw = expected_path.read_bytes()

    # Duplicate key와 semantic-equal reformat을 file SHA보다 구체적으로 차단한다.
    authority = _load_canonical_json(raw, label="live capture authority")
    _exact_keys(authority, _AUTHORITY_KEYS, label="live capture authority")
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("legacy/v4/old broadband live authority schema를 거부합니다")

    actual_file_sha = _file_sha256_bytes(raw)
    if (
        actual_file_sha != external_file_sha
        or actual_file_sha != EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
    ):
        raise ValueError("live capture authority file SHA가 expected/pinned 값과 다릅니다")
    internal_payload_sha = _require_sha256(
        authority.get("authority_sha256"),
        label="saved live capture authority payload SHA",
    )
    core = {
        key: item
        for key, item in authority.items()
        if key != "authority_sha256"
    }
    recomputed_payload_sha = _payload_sha256(core)
    if (
        internal_payload_sha != external_payload_sha
        or internal_payload_sha != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256
        or recomputed_payload_sha != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256
    ):
        raise ValueError("live capture authority payload SHA가 internal/external/pinned 값과 다릅니다")

    validation = validate_live_capture_authority_v5(
        authority,
        repository_root=root,
        expected_authority_sha256=external_payload_sha,
    )
    return {
        "path": SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        "file_sha256": actual_file_sha,
        "payload_sha256": recomputed_payload_sha,
        "authority": authority,
        "validation": validation,
    }


__all__ = [
    "AUTHORITY_SCHEMA",
    "EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256",
    "EXPECTED_HARDWARE_FILE_SHA256",
    "EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256",
    "EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256",
    "EXPECTED_PCM_SHA256",
    "EXPECTED_PLAN_ENVELOPE_FILE_SHA256",
    "EXPECTED_PLAN_PAYLOAD_SHA256",
    "PLAN_ENVELOPE_SCHEMA",
    "SEALED_HARDWARE_RELATIVE_PATH",
    "SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH",
    "SEALED_PLAN_ENVELOPE_RELATIVE_PATH",
    "SEALED_RAW_RELATIVE_PATH",
    "build_live_capture_authority_v5",
    "canonical_plan_envelope_bytes_v5",
    "committed_plan_envelope_v5",
    "load_exact_saved_plan_v5",
    "load_exact_saved_live_capture_authority_v5",
    "validate_live_capture_authority_v5",
]
