"""Fullband-v6 signal plan의 capture-only authority 계약.

v5 authority/assets를 변경하지 않는다.  v6 signal builder는 병렬 작업 중이므로
import를 함수 안에서 수행하고, builder 결과를 실제 int16 PCM에서 다시 검산한다.
이 모듈은 오디오 backend를 import하지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from deep_anc.data.repository_fd import RepositoryFileGuard, repository_root as _repository_root

FS = 48_000
BLOCK = 256
PERIOD = 32_768
CLOCK_PREFIX = PERIOD // 2
CLOCK_SUFFIX = PERIOD // 2
CLOCK_BLOCK_FRAMES = 3 * PERIOD
CLOCK_BLOCKS = 8
PE_BLOCK_FRAMES = 2 * PERIOD
PE_BLOCKS = 6
TOTAL_FRAMES = 36 * PERIOD
DURATION_SECONDS = TOTAL_FRAMES / FS
FIXED_CLOCK_BINS = (109, 137, 181, 233, 277, 314, 359, 401)
SUBMITTED_PEAK_LIMIT_PCM = 98
METER_RELATIVE_POWER_DB = (-0.25, 0.0)

PLAN_SCHEMA = "fullband_causal_time_separated_clock_v6"
PLAN_ENVELOPE_SCHEMA = "fullband_causal_signal_plan_envelope_v6"
AUTHORITY_SCHEMA = "fullband_causal_v6_live_capture_authority_v1"
SEALED_PLAN_ENVELOPE_RELATIVE_PATH = (
    "assets/contracts/fullband_causal_v6_signal_plan.json"
)
SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH = (
    "assets/contracts/fullband_causal_v6_live_capture_authority.json"
)
SEALED_RAW_RELATIVE_PATH = "results/fullband_causal_v6/raw_capture.npz"
SEALED_HARDWARE_RELATIVE_PATH = "configs/hardware_jetson.yaml"
EXPECTED_PLAN_PAYLOAD_SHA256 = "8b37213a13131a071e10527c948580c906dfd914a1134e98a640ead259ba42f7"
EXPECTED_PCM_SHA256 = "4e8a66b983af872192624bd6759282058cfe4a845460111a24bcd684b22551a3"
EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256 = "211f581296d9d99927241a08c7a1096615246d68fe6702db8ff241cf1f582034"
EXPECTED_PLAN_ENVELOPE_PAYLOAD_SHA256 = "4ddd9df15469a288577bbbeac87091decd0ac2954d4e02f377968512dde52e40"
EXPECTED_PLAN_ENVELOPE_FILE_SHA256 = "500b93d1a5289ac0d467683088ea2d72181810f45872faf0bcb29265bb13cf3b"
EXPECTED_HARDWARE_FILE_SHA256 = "45232a45e51fd76c7b88db338b9cf4f3840a88299b4d452e259064c0ee559351"
EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256 = "136728b0dab2c0068adbbb21da7e0a5fa315d1e036d3afa534b647f85d6e0f32"
EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256 = "7a795e4e780004d4260fd85abab5c73e6d46858b3ab99c551997a2337fd15b75"


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pcm_sha256(pcm: np.ndarray) -> str:
    value = np.ascontiguousarray(pcm, dtype="<i2")
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def design_contract_v6() -> dict[str, Any]:
    core = {
        "schema": "fullband_causal_v6_design_contract_v1",
        "sample_rate": FS,
        "block_size": BLOCK,
        "period_frames": PERIOD,
        "clock": {
            "prefix_frames": CLOCK_PREFIX,
            "central_periods": 2,
            "suffix_frames": CLOCK_SUFFIX,
            "block_frames": CLOCK_BLOCK_FRAMES,
            "blocks": CLOCK_BLOCKS,
            "fixed_bins": list(FIXED_CLOCK_BINS),
            "active_path_only": True,
            "opposite_path_exact_zero": True,
        },
        "pe": {"block_frames": PE_BLOCK_FRAMES, "blocks": PE_BLOCKS},
        "total_frames": TOTAL_FRAMES,
        "duration_seconds": DURATION_SECONDS,
        "actual_int16_meter_relative_power_db": list(METER_RELATIVE_POWER_DB),
        "submitted_peak_limit_pcm": SUBMITTED_PEAK_LIMIT_PCM,
    }
    return {**core, "contract_sha256": payload_sha256(core)}


def _builder_result() -> tuple[dict[str, Any], np.ndarray]:
    module = importlib.import_module("deep_anc.dsp.fullband_causal_v6")
    builder = getattr(module, "build_plan_v6", None)
    if not callable(builder):
        raise RuntimeError("fullband_causal_v6.build_plan_v6가 필요합니다")
    plan, submitted = builder(raw_session_relative_path=SEALED_RAW_RELATIVE_PATH)
    if not isinstance(plan, dict):
        raise ValueError("v6 builder plan은 dict여야 합니다")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("v6 builder plan schema가 final contract와 다릅니다")
    pcm = np.asarray(submitted)
    if pcm.dtype != np.dtype("int16") or pcm.shape != (TOTAL_FRAMES, 2):
        raise ValueError("v6 submitted PCM은 exact int16 [1179648,2]여야 합니다")
    if not pcm.flags.c_contiguous:
        raise ValueError("v6 submitted PCM은 C-contiguous여야 합니다")
    if int(np.max(np.abs(pcm.astype(np.int32)))) > SUBMITTED_PEAK_LIMIT_PCM:
        raise ValueError("v6 submitted PCM peak가 98을 초과합니다")
    if TOTAL_FRAMES % BLOCK:
        raise AssertionError("v6 total frame이 block aligned가 아닙니다")
    declared = plan.get("actual_submitted_pcm_sha256")
    actual = pcm_sha256(pcm)
    if declared != actual:
        raise ValueError("v6 plan과 actual int16 PCM SHA가 다릅니다")
    if plan.get("canonical_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256 or actual != EXPECTED_PCM_SHA256:
        raise ValueError("v6 builder plan/PCM이 pinned SHA와 다릅니다")
    if plan.get("raw_session_relative_path") != SEALED_RAW_RELATIVE_PATH or plan.get("publisher_contract", {}).get("raw_session_relative_path") != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("v6 sealed raw path가 builder와 다릅니다")
    return plan, np.array(pcm, dtype="<i2", copy=True, order="C")


@lru_cache(maxsize=1)
def _expected_condition_receipt_v6() -> dict[str, Any]:
    plan, submitted = _builder_result()
    module = importlib.import_module("deep_anc.dsp.fullband_causal_v6")
    audit = getattr(module, "exact_condition_audit_v6", None)
    if not callable(audit):
        raise RuntimeError("fullband_causal_v6.exact_condition_audit_v6가 필요합니다")
    receipt = audit(plan, submitted)
    if not isinstance(receipt, dict):
        raise ValueError("v6 condition receipt는 dict여야 합니다")
    declared = receipt.get("canonical_payload_sha256")
    core = {key: item for key, item in receipt.items() if key != "canonical_payload_sha256"}
    if (
        declared != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        or payload_sha256(core) != EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
        or receipt.get("schema") != "fullband_causal_exact_gram_condition_v6"
        or receipt.get("support_samples") != 1024
        or receipt.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
        or receipt.get("signal_plan_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256
        or receipt.get("fit_roles") != ["fit_a", "fit_b"]
        or receipt.get("passed") is not True
        or float(receipt.get("joint_fit_condition_number", float("inf"))) > 20.0
    ):
        raise ValueError("v6 support-1024 condition receipt exact SHA/PASS gate 실패")
    return json.loads(canonical_json_bytes(receipt).decode("utf-8"))


def committed_plan_envelope_v6() -> dict[str, Any]:
    plan, submitted = _builder_result()
    contract = design_contract_v6()
    receipt = _expected_condition_receipt_v6()
    if (
        receipt.get("actual_submitted_pcm_sha256") != pcm_sha256(submitted)
        or receipt.get("signal_plan_payload_sha256") != plan.get("canonical_payload_sha256")
    ):
        raise ValueError("cached v6 condition receipt가 current plan/PCM과 다릅니다")
    return {
        "schema": PLAN_ENVELOPE_SCHEMA,
        "design_contract": contract,
        "signal_plan": plan,
        "actual_submitted_pcm_sha256": pcm_sha256(submitted),
        "support_1024_condition_receipt": receipt,
    }


def canonical_plan_envelope_bytes_v6() -> bytes:
    raw = canonical_json_bytes(committed_plan_envelope_v6(), pretty=True)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise AssertionError("v6 plan envelope file SHA가 pinned 값과 다릅니다")
    return raw


def build_live_capture_authority_v6(
    *, plan_envelope_file_sha256: str, hardware_file_sha256: str
) -> dict[str, Any]:
    for value, label in (
        (plan_envelope_file_sha256, "plan envelope"),
        (hardware_file_sha256, "hardware"),
    ):
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{label} SHA-256이 유효하지 않습니다")
    if plan_envelope_file_sha256 != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise ValueError("v6 plan envelope SHA가 pinned 값과 다릅니다")
    if hardware_file_sha256 != EXPECTED_HARDWARE_FILE_SHA256:
        raise ValueError("v6 hardware SHA가 pinned 값과 다릅니다")
    envelope = committed_plan_envelope_v6()
    envelope_payload_sha256 = payload_sha256(envelope)
    if envelope_payload_sha256 != EXPECTED_PLAN_ENVELOPE_PAYLOAD_SHA256:
        raise AssertionError("v6 plan envelope payload SHA가 pinned 값과 다릅니다")
    core = {
        "schema": AUTHORITY_SCHEMA,
        "scope": "capture_only_not_plant_delay_or_training_authority",
        "capture_only": True,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "live_delay_authority": None,
        "signal_plan_envelope": {
            "path": SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": plan_envelope_file_sha256,
            "payload_sha256": envelope_payload_sha256,
            "pcm_sha256": envelope["actual_submitted_pcm_sha256"],
            "condition_receipt_payload_sha256": EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256,
        },
        "hardware": {
            "path": SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": hardware_file_sha256,
        },
        "sealed_raw": {
            "path": SEALED_RAW_RELATIVE_PATH,
            "must_not_exist_before_capture": True,
        },
    }
    result = {**core, "authority_sha256": payload_sha256(core)}
    if result["authority_sha256"] != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256:
        raise AssertionError("v6 authority payload SHA가 pinned 값과 다릅니다")
    return result


def canonical_live_capture_authority_bytes_v6() -> bytes:
    raw = canonical_json_bytes(
        build_live_capture_authority_v6(
            plan_envelope_file_sha256=EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            hardware_file_sha256=EXPECTED_HARDWARE_FILE_SHA256,
        ),
        pretty=True,
    )
    if hashlib.sha256(raw).hexdigest() != EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256:
        raise AssertionError("v6 authority file SHA가 pinned 값과 다릅니다")
    return raw


def asset_payloads_v6(*, hardware_file_sha256: str) -> dict[str, bytes]:
    """검토 후 no-replace publisher에 넘길 두 canonical asset bytes를 만든다."""

    if hardware_file_sha256 != EXPECTED_HARDWARE_FILE_SHA256:
        raise ValueError("v6 asset hardware SHA가 pinned 값과 다릅니다")
    envelope_bytes = canonical_plan_envelope_bytes_v6()
    return {
        SEALED_PLAN_ENVELOPE_RELATIVE_PATH: envelope_bytes,
        SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH: canonical_live_capture_authority_bytes_v6(),
    }


def _load_exact_file(root: Path, relative: str, expected_sha: str) -> bytes:
    with RepositoryFileGuard(root, relative, label="v6 sealed asset") as guard:
        if guard.sha256 != expected_sha:
            raise ValueError("v6 sealed asset file SHA가 pinned 값과 다릅니다")
        guard.verify()
        return guard.bytes


def load_exact_saved_plan_v6(
    plan_envelope_path: str | os.PathLike[str], *, repository_root: str | os.PathLike[str],
    expected_file_sha256: str,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    expected = root / SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    supplied = Path(os.path.abspath(os.fspath(plan_envelope_path)))
    if supplied != expected or expected_file_sha256 != EXPECTED_PLAN_ENVELOPE_FILE_SHA256:
        raise ValueError("v6 plan path/expected SHA가 sealed 값과 다릅니다")
    raw = _load_exact_file(root, SEALED_PLAN_ENVELOPE_RELATIVE_PATH, expected_file_sha256)
    if raw != canonical_plan_envelope_bytes_v6():
        raise ValueError("v6 plan canonical bytes가 builder와 다릅니다")
    envelope = json.loads(raw)
    return {"path": SEALED_PLAN_ENVELOPE_RELATIVE_PATH, "file_sha256": expected_file_sha256,
            "payload_sha256": EXPECTED_PLAN_PAYLOAD_SHA256, "pcm_sha256": EXPECTED_PCM_SHA256,
            "condition_receipt_payload_sha256": EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256,
            "sealed_raw_relative_path": SEALED_RAW_RELATIVE_PATH, "envelope": envelope}


def load_exact_saved_live_capture_authority_v6(
    authority_path: str | os.PathLike[str], *, repository_root: str | os.PathLike[str],
    expected_file_sha256: str, expected_payload_sha256: str,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    expected = root / SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
    supplied = Path(os.path.abspath(os.fspath(authority_path)))
    if supplied != expected or expected_file_sha256 != EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256 or expected_payload_sha256 != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256:
        raise ValueError("v6 authority path/expected SHA가 sealed 값과 다릅니다")
    raw = _load_exact_file(root, SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH, expected_file_sha256)
    if raw != canonical_live_capture_authority_bytes_v6():
        raise ValueError("v6 authority canonical bytes가 builder와 다릅니다")
    value = validate_live_capture_authority_v6(json.loads(raw))
    return {"path": SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH, "file_sha256": expected_file_sha256,
            "payload_sha256": expected_payload_sha256, "authority": value}


def validate_live_capture_authority_v6(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "scope", "capture_only", "canonical_training_eligible",
        "hardware_sample_slip_authority", "live_delay_authority",
        "signal_plan_envelope", "hardware", "sealed_raw", "authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("v6 authority key 집합이 exact하지 않습니다")
    core = {key: item for key, item in value.items() if key != "authority_sha256"}
    if value.get("authority_sha256") != payload_sha256(core):
        raise ValueError("v6 authority payload SHA가 다릅니다")
    expected_core = {
        "schema": AUTHORITY_SCHEMA,
        "scope": "capture_only_not_plant_delay_or_training_authority",
        "capture_only": True,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "live_delay_authority": None,
        "signal_plan_envelope": {
            "path": SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            "payload_sha256": EXPECTED_PLAN_ENVELOPE_PAYLOAD_SHA256,
            "pcm_sha256": EXPECTED_PCM_SHA256,
            "condition_receipt_payload_sha256": EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256,
        },
        "hardware": {
            "path": SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": EXPECTED_HARDWARE_FILE_SHA256,
        },
        "sealed_raw": {
            "path": SEALED_RAW_RELATIVE_PATH,
            "must_not_exist_before_capture": True,
        },
    }
    if core != expected_core:
        raise ValueError("v6 authority가 exact pinned capture-only 계약과 다릅니다")
    if value["authority_sha256"] != EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256:
        raise ValueError("v6 authority SHA가 pinned authority와 다릅니다")
    PurePosixPath(SEALED_RAW_RELATIVE_PATH)  # lexical constant witness
    return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))


__all__ = [name for name in globals() if name.isupper()] + [
    "asset_payloads_v6", "build_live_capture_authority_v6", "canonical_json_bytes",
    "canonical_plan_envelope_bytes_v6", "canonical_live_capture_authority_bytes_v6",
    "committed_plan_envelope_v6", "design_contract_v6", "payload_sha256",
    "load_exact_saved_plan_v6", "load_exact_saved_live_capture_authority_v6",
    "pcm_sha256", "validate_live_capture_authority_v6",
]
