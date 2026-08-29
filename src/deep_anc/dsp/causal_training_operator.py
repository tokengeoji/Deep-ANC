"""v4 joint causal P/S training operator의 단일 byte loader.

tone-only measured response나 source 전용 derived P NPZ를 이 역할로 승격하지 않는다.
P/S는 한 NPZ의 같은 generation에 함께 있고, criterion과 source-v2가 이 모듈에서
각 역할 view를 얻는다. 별도 P 복제본을 만들지 않아 FIR SHA drift를 구조적으로 막는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import fullband_causal_v4 as v4
from .control_band_contract import ControlBandContract
from .fullband_causal_v4 import OPERATOR_NPZ_SCHEMA
from .timing import PlantDelays, TrainingTimingContract


NPZ_INTERNAL_DIGEST_DOMAIN = "joint_causal_operator_npz_internal_v4"
EXACT_CLOCK_CHANGE_POINT_VALIDATOR_AVAILABLE = False
_HEX = frozenset("0123456789abcdef")
_NPZ_KEYS = {
    "schema",
    "primary_post_onset_fir",
    "secondary_post_onset_fir",
    "primary_coarse_delay_samples",
    "secondary_coarse_delay_samples",
    "primary_fractional_delay_samples",
    "secondary_fractional_delay_samples",
    "support_samples",
    "sample_rate_hz",
    "source_submitted_pcm_sha256",
    "source_raw_sha256",
    "fit_freeze_sha256",
}
_AUTHORITY_KEYS = {
    "schema", "authority", "status", "canonical_training_eligible",
    "synthetic_fixture", "control_band_contract_sha256", "sample_rate_hz",
    "block_size", "latency", "handoff_extra_samples", "capture_id",
    "operator", "clock", "fit", "holdout", "stationarity", "provenance",
    "evidence_sha256",
}
_OPERATOR_REFERENCE_KEYS = {
    "schema", "npz_path", "npz_file_sha256", "npz_internal_sha256",
    "primary_fir_sha256", "secondary_fir_sha256",
    "source_submitted_pcm_sha256", "source_raw_sha256", "fit_freeze_sha256",
    "support_samples", "coarse_delay_samples", "fractional_delay_samples",
    "bulk_delay_samples_fractional", "post_onset_peak_index",
    "effective_delay_samples", "plant_delays_payload", "plant_delays_sha256",
}
_FILE_IDENTITY_REFERENCE_KEYS = {"schema", "path", "file_sha256", "internal_sha256"}
_FIT_ROLE_KEYS = {
    "fit_a_candidate_ref", "fit_b_candidate_ref", "fit_a_score", "fit_b_score",
    "fit_a_on_fit_b_score", "fit_b_on_fit_a_score", "freeze_ref",
    "selected_support_samples",
}
_PROVENANCE_KEYS = {
    "repository_commit", "repository_dirty", "signal_plan_path",
    "signal_plan_file_sha256", "signal_plan_payload_sha256",
    "submitted_pcm_path", "submitted_pcm_file_sha256", "raw_path",
    "raw_file_sha256", "raw_internal_sha256", "callback_arrays_sha256",
    "analysis_path", "analysis_file_sha256", "analysis_internal_sha256",
    "analysis_code_sha256", "environment_receipt_sha256",
    "level_evidence_sha256", "hardware_fingerprint_sha256", "xrun_count",
    "clip_count",
}
_CALLBACK_ARRAY_KEYS = {
    "callback_start_frames", "callback_frame_counts", "input_buffer_adc_time",
    "output_buffer_dac_time", "callback_current_time",
}


class CausalTrainingAuthorityUnavailable(RuntimeError):
    """구조 parser가 있어도 reviewed live authority가 아직 없음을 나타낸다."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact(value: object, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 exact하지 않습니다: {actual}")
    return value


def _json_bytes_no_duplicates(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON을 안전하게 읽을 수 없습니다: {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위가 object가 아닙니다: {label}")
    return value


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    return _json_bytes_no_duplicates(path.read_bytes(), label=str(path))


def _relative_regular_file(base: Path, value: object, *, label: str) -> Path:
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}는 authority 기준 lexical relative path여야 합니다")
    path = Path(os.path.abspath(base / relative))
    cursor = base
    for component in relative.parts:
        cursor /= component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} regular file이 없습니다: {path}")
    return path


def _sealed_receipt(value: object, *, schema: str | None, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}가 mapping이 아닙니다")
    payload = dict(value)
    claimed = require_sha256(payload.pop("receipt_sha256", None), label=f"{label} SHA")
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(f"{label} schema가 다릅니다")
    if claimed != _json_sha256(payload):
        raise ValueError(f"{label} receipt SHA 재계산이 다릅니다")
    result = dict(payload)
    result["receipt_sha256"] = claimed
    return result


def _archive_digest(
    arrays: dict[str, np.ndarray], *, domain: str, selected: set[str] | None = None
) -> str:
    keys = set(arrays) if selected is None else set(selected)
    if not keys.issubset(arrays):
        raise ValueError(f"{domain} digest 대상 array가 누락됐습니다")
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8") + b"\0")
    for key in sorted(keys):
        array = np.asarray(arrays[key])
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("utf-8") + b"\0")
        digest.update(np.asarray(array.ndim, dtype="<i8").tobytes())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_regular_file(
    repository_root: str | Path, value: str | Path, *, label: str
) -> Path:
    root = Path(os.path.abspath(Path(repository_root).expanduser()))
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} repository root가 유효하지 않습니다")
    raw = Path(value).expanduser()
    path = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}가 repository 밖입니다") from exc
    cursor = root
    for component in relative.parts:
        cursor /= component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} regular file이 없습니다: {path}")
    return path


def _utf8_vector(value: np.ndarray, *, label: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype("uint8") or array.ndim != 1 or array.size < 1:
        raise ValueError(f"{label}는 canonical uint8 UTF-8 vector여야 합니다")
    try:
        text = array.tobytes(order="C").decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}가 유효한 UTF-8이 아닙니다") from exc
    if not text or text.encode("utf-8") != array.tobytes(order="C"):
        raise ValueError(f"{label} UTF-8 canonical round-trip이 다릅니다")
    return text


def operator_npz_internal_sha256(arrays: dict[str, np.ndarray]) -> str:
    if set(arrays) != _NPZ_KEYS:
        raise ValueError("joint causal operator NPZ key 집합이 exact하지 않습니다")
    digest = hashlib.sha256()
    digest.update(NPZ_INTERNAL_DIGEST_DOMAIN.encode("utf-8") + b"\0")
    for key in sorted(arrays):
        array = np.asarray(arrays[key])
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("utf-8") + b"\0")
        digest.update(np.asarray(array.ndim, dtype="<i8").tobytes())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _int64_scalar(value: np.ndarray, *, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("<i8"):
        raise ValueError(f"{label}는 little-endian int64 scalar여야 합니다")
    return int(array.item())


def _float64_scalar(value: np.ndarray, *, label: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("<f8"):
        raise ValueError(f"{label}는 little-endian float64 scalar여야 합니다")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{label}가 finite가 아닙니다")
    return result


@dataclass(frozen=True)
class JointCausalOperatorData:
    path: Path
    file_sha256: str
    internal_sha256: str
    primary_post_onset_fir: np.ndarray
    secondary_post_onset_fir: np.ndarray
    primary_coarse_delay_samples: int
    secondary_coarse_delay_samples: int
    primary_fractional_delay_samples: float
    secondary_fractional_delay_samples: float
    support_samples: int
    sample_rate_hz: int
    source_submitted_pcm_sha256: str
    source_raw_sha256: str
    fit_freeze_sha256: str
    primary_fir_sha256: str
    secondary_fir_sha256: str


def load_joint_causal_operator_npz(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_internal_sha256: str,
) -> JointCausalOperatorData:
    """한 FD generation의 exact P/S arrays와 digest를 검증해 반환한다."""

    operator_path = Path(path)
    expected_file = require_sha256(
        expected_file_sha256, label="joint causal operator file SHA"
    )
    expected_internal = require_sha256(
        expected_internal_sha256, label="joint causal operator internal SHA"
    )
    raw = operator_path.read_bytes()
    actual_file = hashlib.sha256(raw).hexdigest()
    if actual_file != expected_file:
        raise ValueError("joint causal operator NPZ file bytes가 authority와 다릅니다")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_KEYS:
                raise ValueError("joint causal operator NPZ key 집합이 exact하지 않습니다")
            arrays = {
                key: np.array(archive[key], copy=True, order="C")
                for key in archive.files
            }
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("joint causal operator NPZ를 안전하게 읽을 수 없습니다") from exc
    actual_internal = operator_npz_internal_sha256(arrays)
    if actual_internal != expected_internal:
        raise ValueError("joint causal operator NPZ internal digest가 authority와 다릅니다")
    if _utf8_vector(arrays["schema"], label="operator schema") != OPERATOR_NPZ_SCHEMA:
        raise ValueError("joint causal operator NPZ schema가 v4가 아닙니다")

    support = _int64_scalar(arrays["support_samples"], label="support_samples")
    sample_rate = _int64_scalar(arrays["sample_rate_hz"], label="sample_rate_hz")
    primary = np.asarray(arrays["primary_post_onset_fir"])
    secondary = np.asarray(arrays["secondary_post_onset_fir"])
    if (
        primary.dtype != np.dtype("<f8")
        or secondary.dtype != np.dtype("<f8")
        or primary.ndim != 1
        or secondary.ndim != 1
        or primary.shape != (support,)
        or secondary.shape != (support,)
        or support < 1
        or sample_rate != 48_000
        or not np.all(np.isfinite(primary))
        or not np.all(np.isfinite(secondary))
        or float(np.max(np.abs(primary))) <= 0.0
        or float(np.max(np.abs(secondary))) <= 0.0
    ):
        raise ValueError("joint causal operator P/S float64 support/fs 계약 위반")
    primary_delay = _int64_scalar(
        arrays["primary_coarse_delay_samples"], label="primary coarse delay"
    )
    secondary_delay = _int64_scalar(
        arrays["secondary_coarse_delay_samples"], label="secondary coarse delay"
    )
    primary_fractional = _float64_scalar(
        arrays["primary_fractional_delay_samples"], label="primary fractional delay"
    )
    secondary_fractional = _float64_scalar(
        arrays["secondary_fractional_delay_samples"], label="secondary fractional delay"
    )
    if (
        primary_delay < 0
        or secondary_delay < 0
        or not -0.5 <= primary_fractional < 0.5
        or not -0.5 <= secondary_fractional < 0.5
    ):
        raise ValueError("joint causal operator coarse/fractional delay 계약 위반")
    submitted_sha = require_sha256(
        _utf8_vector(
            arrays["source_submitted_pcm_sha256"], label="submitted PCM SHA"
        ),
        label="submitted PCM SHA",
    )
    raw_sha = require_sha256(
        _utf8_vector(arrays["source_raw_sha256"], label="source raw SHA"),
        label="source raw SHA",
    )
    freeze_sha = require_sha256(
        _utf8_vector(arrays["fit_freeze_sha256"], label="fit freeze SHA"),
        label="fit freeze SHA",
    )
    primary = np.ascontiguousarray(primary, dtype="<f8")
    secondary = np.ascontiguousarray(secondary, dtype="<f8")
    return JointCausalOperatorData(
        path=operator_path,
        file_sha256=actual_file,
        internal_sha256=actual_internal,
        primary_post_onset_fir=primary,
        secondary_post_onset_fir=secondary,
        primary_coarse_delay_samples=primary_delay,
        secondary_coarse_delay_samples=secondary_delay,
        primary_fractional_delay_samples=primary_fractional,
        secondary_fractional_delay_samples=secondary_fractional,
        support_samples=support,
        sample_rate_hz=sample_rate,
        source_submitted_pcm_sha256=submitted_sha,
        source_raw_sha256=raw_sha,
        fit_freeze_sha256=freeze_sha,
        primary_fir_sha256=hashlib.sha256(primary.tobytes(order="C")).hexdigest(),
        secondary_fir_sha256=hashlib.sha256(secondary.tobytes(order="C")).hexdigest(),
    )


def _load_identity_json(
    raw: object, *, authority_dir: Path, expected_schema: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    reference = _exact(raw, _FILE_IDENTITY_REFERENCE_KEYS, label=label)
    if reference["schema"] != v4.IMMUTABLE_JSON_ARTIFACT_REFERENCE_SCHEMA:
        raise ValueError(f"{label} reference schema가 다릅니다")
    path = _relative_regular_file(authority_dir, reference["path"], label=label)
    file_sha = require_sha256(reference["file_sha256"], label=f"{label} file SHA")
    file_bytes = path.read_bytes()
    if hashlib.sha256(file_bytes).hexdigest() != file_sha:
        raise ValueError(f"{label} file bytes가 authority와 다릅니다")
    payload = _json_bytes_no_duplicates(file_bytes, label=str(path))
    identity_key = (
        "candidate_sha256"
        if expected_schema == "joint_actual_input_fit_candidate_v4"
        else "freeze_sha256"
    )
    claimed = require_sha256(payload.get(identity_key), label=f"{label} internal SHA")
    if claimed != _json_sha256(
        {key: value for key, value in payload.items() if key != identity_key}
    ):
        raise ValueError(f"{label} JSON internal SHA 재계산이 다릅니다")
    if claimed != require_sha256(
        reference["internal_sha256"], label=f"{label} reference internal SHA"
    ):
        raise ValueError(f"{label} reference/internal identity가 다릅니다")
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{label} JSON schema가 다릅니다")
    return payload, path, claimed


def _load_submitted_pcm(path: Path, *, file_bytes: bytes | None = None) -> np.ndarray:
    """no-replace ``.npy``의 exact C-contiguous stereo int16을 읽는다."""

    if path.suffix != ".npy":
        raise ValueError("submitted PCM은 no-replace .npy여야 합니다")
    try:
        value = np.load(
            io.BytesIO(path.read_bytes() if file_bytes is None else file_bytes),
            allow_pickle=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("submitted PCM .npy를 안전하게 읽을 수 없습니다") from exc
    array = np.asarray(value)
    if (
        array.dtype != np.dtype("<i2")
        or array.ndim != 2
        or array.shape[1] != 2
        or array.shape[0] < 1
        or not array.flags.c_contiguous
    ):
        raise ValueError("submitted PCM은 C-contiguous [frames,2] int16이어야 합니다")
    return array


def _candidate_subbands(
    candidate: dict[str, Any], *, contract_sha256: str, label: str
) -> tuple[tuple[float, float], ...]:
    if candidate.get("control_band_contract_sha256") != contract_sha256:
        raise ValueError(f"{label} control-band contract SHA가 authority와 다릅니다")
    try:
        bands = tuple(
            (float(value[0]), float(value[1]))
            for value in candidate["control_subbands_hz"]
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{label} declared subband가 유효하지 않습니다") from exc
    if (
        not bands
        or any(
            not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi)
            for lo, hi in bands
        )
        or any(abs(bands[index - 1][1] - bands[index][0]) > 1.0e-9 for index in range(1, len(bands)))
    ):
        raise ValueError(f"{label} declared subband union이 유효하지 않습니다")
    return bands


def _validate_score_rows(
    score: dict[str, Any],
    *,
    subbands_hz: tuple[tuple[float, float], ...],
    label: str,
) -> None:
    expected = {
        (path, index)
        for path in ("primary", "secondary")
        for index in range(len(subbands_hz))
    }
    rows = score.get("subband_rows")
    if not isinstance(rows, list) or {
        (str(row.get("path")), int(row.get("band_index", -1)))
        for row in rows
        if isinstance(row, dict)
    } != expected:
        raise ValueError(f"{label}에 P/S×모든 contract subband row가 없습니다")
    for row in rows:
        path = str(row["path"])
        index = int(row["band_index"])
        band = [float(value) for value in subbands_hz[index]]
        if [float(value) for value in row.get("band_hz", [])] != band:
            raise ValueError(f"{label} {path}/{index} band가 contract와 다릅니다")
        timing_limit = float(row.get("max_abs_phase_delay_samples", -1.0))
        numeric_pass = bool(
            int(row.get("isolated_response_bin_count", -1))
            >= int(v4.SUBBAND_MIN_RESPONSE_BINS)
            and int(row.get("phase_bin_count", -1))
            >= int(v4.SUBBAND_MIN_RESPONSE_BINS)
            and int(row.get("exact_zero_noise_bin_count", -1))
            >= int(v4.SUBBAND_MIN_EXACT_ZERO_NOISE_BINS)
            and float(row.get("input_rms_dbfs", -math.inf))
            >= float(v4.SUBBAND_MIN_INPUT_RMS_DBFS)
            and float(row.get("target_rms_dbfs", -math.inf))
            >= float(v4.SUBBAND_MIN_TARGET_RMS_DBFS)
            and float(row.get("target_to_noise_db", -math.inf))
            >= float(v4.SUBBAND_MIN_TARGET_TO_NOISE_DB)
            and float(row.get("noise_conditioned_relative_residual", math.inf))
            <= float(v4.SUBBAND_MAX_RELATIVE_ERROR)
            and float(row.get("complex_agreement", -math.inf))
            >= float(v4.SUBBAND_MIN_COMPLEX_AGREEMENT)
            and float(row.get("phase_coherence", -math.inf))
            >= float(v4.SUBBAND_MIN_COMPLEX_AGREEMENT)
            and timing_limit >= 0.0
            and abs(float(row.get("phase_delay_samples", math.inf))) <= timing_limit
        )
        if row.get("passed") is not True or not numeric_pass:
            raise ValueError(f"{label} {path}/{index} score gate가 FAIL입니다")


def _validate_score(
    raw: object,
    *,
    candidate: dict[str, Any],
    target_role: str,
    microphone_role: str,
    contract_sha256: str,
    label: str,
) -> dict[str, Any]:
    score = _sealed_receipt(
        raw, schema="joint_actual_input_role_score_v4", label=label
    )
    # 생성기 자체의 identity/role/P×S row 검증과 별도로 숫자 threshold를 다시 계산한다.
    v4._validate_score_receipt_v4(  # noqa: SLF001 - 같은 schema authority 재사용
        score,
        candidate=candidate,
        target_role=target_role,
        microphone_role=microphone_role,
        require_pass=True,
    )
    subbands = _candidate_subbands(
        candidate, contract_sha256=contract_sha256, label=label
    )
    _validate_score_rows(score, subbands_hz=subbands, label=label)
    if (
        score.get("control_band_contract_sha256")
        != contract_sha256
        or score.get("global_residual_is_not_sufficient") is not True
        or score.get("all_subbands_passed") is not True
        or score.get("passed") is not True
        or float(score.get("global_residual_ratio", math.inf))
        > float(score.get("global_residual_threshold", -math.inf))
    ):
        raise ValueError(f"{label} global/subband gate가 PASS가 아닙니다")
    noise = _sealed_receipt(
        score.get("noise_floor_receipt"),
        schema="actual_exact_zero_noise_floor_v4",
        label=f"{label}.noise_floor",
    )
    if noise.get("passed") is not True:
        raise ValueError(f"{label} exact-zero noise-floor gate가 FAIL입니다")
    return score


def _validate_stationarity(
    raw: object,
    *,
    fit_a_sha: str,
    fit_b_sha: str,
    contract_sha256: str,
    subbands_hz: tuple[tuple[float, float], ...],
    label: str,
) -> dict[str, Any]:
    receipt = _sealed_receipt(
        raw, schema="fit_a_fit_b_transfer_stationarity_v4", label=label
    )
    if (
        receipt.get("fit_a_candidate_sha256") != fit_a_sha
        or receipt.get("fit_b_candidate_sha256") != fit_b_sha
        or receipt.get("control_band_contract_sha256") != contract_sha256
        or receipt.get("all_subbands_passed") is not True
        or receipt.get("passed") is not True
    ):
        raise ValueError(f"{label} candidate identity/PASS가 다릅니다")
    rows = receipt.get("subband_rows")
    expected = {
        (path, index)
        for path in ("primary", "secondary")
        for index in range(len(subbands_hz))
    }
    observed = {
        (str(row.get("path")), int(row.get("band_index", -1)))
        for row in rows
        if isinstance(row, dict)
    } if isinstance(rows, list) else set()
    if (
        observed != expected
        or any(row.get("passed") is not True for row in rows or [])
        or any(
            tuple(float(value) for value in row.get("band_hz", ()))
            != subbands_hz[int(row["band_index"])]
            for row in rows or []
        )
    ):
        raise ValueError(f"{label} P/S×contract band stationarity가 누락/실패했습니다")
    return receipt


def _validate_fit_role(
    raw: object,
    *,
    authority_dir: Path,
    microphone_role: str,
    submitted_sha: str,
    raw_sha: str,
    contract_sha256: str,
) -> dict[str, Any]:
    role = _exact(raw, _FIT_ROLE_KEYS, label=f"fit.{microphone_role}")
    fit_a, fit_a_path, fit_a_sha = _load_identity_json(
        role["fit_a_candidate_ref"],
        authority_dir=authority_dir,
        expected_schema="joint_actual_input_fit_candidate_v4",
        label=f"fit.{microphone_role}.fit_a",
    )
    fit_b, fit_b_path, fit_b_sha = _load_identity_json(
        role["fit_b_candidate_ref"],
        authority_dir=authority_dir,
        expected_schema="joint_actual_input_fit_candidate_v4",
        label=f"fit.{microphone_role}.fit_b",
    )
    for name, candidate, source_role in (
        ("fit_a", fit_a, "fit_a"),
        ("fit_b", fit_b, "fit_b"),
    ):
        if (
            candidate.get("microphone_role") != microphone_role
            or candidate.get("source_role") != source_role
            or candidate.get("submitted_pcm_sha256") != submitted_sha
            or candidate.get("canonical_training_eligible") is not False
            or candidate.get("fractional_delay_encoded_in_post_onset_fir") is not True
        ):
            raise ValueError(f"fit.{microphone_role}.{name} lineage/role가 다릅니다")
        # response raw lineage는 analysis artifact에 재결속되어야 한다.
        require_sha256(candidate.get("response_sha256"), label=f"{name} response SHA")
        require_sha256(candidate.get("selected_indices_sha256"), label=f"{name} indices SHA")
        if int(candidate.get("support_samples", -1)) != int(role["selected_support_samples"]):
            raise ValueError(f"fit.{microphone_role}.{name} support가 다릅니다")
    fit_a_subbands = _candidate_subbands(
        fit_a, contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_a",
    )
    fit_b_subbands = _candidate_subbands(
        fit_b, contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_b",
    )
    if fit_a_subbands != fit_b_subbands:
        raise ValueError(f"fit.{microphone_role} fit_a/fit_b subband가 다릅니다")
    fit_a_score = _validate_score(
        role["fit_a_score"], candidate=fit_a, target_role="fit_a",
        microphone_role=microphone_role, contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_a_score",
    )
    fit_b_score = _validate_score(
        role["fit_b_score"], candidate=fit_b, target_role="fit_b",
        microphone_role=microphone_role, contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_b_score",
    )
    cross_a = _validate_score(
        role["fit_a_on_fit_b_score"], candidate=fit_a, target_role="fit_b",
        microphone_role=microphone_role,
        contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_a_on_fit_b",
    )
    cross_b = _validate_score(
        role["fit_b_on_fit_a_score"], candidate=fit_b, target_role="fit_a",
        microphone_role=microphone_role,
        contract_sha256=contract_sha256,
        label=f"fit.{microphone_role}.fit_b_on_fit_a",
    )
    freeze, freeze_path, freeze_sha = _load_identity_json(
        role["freeze_ref"],
        authority_dir=authority_dir,
        expected_schema="frozen_fit_only_joint_causal_candidate_v4",
        label=f"fit.{microphone_role}.freeze",
    )
    if (
        freeze.get("fit_candidate_sha256") != [fit_a_sha, fit_b_sha]
        or int(freeze.get("support_samples", -1)) != int(role["selected_support_samples"])
        or freeze.get("submitted_pcm_sha256") != submitted_sha
        or freeze.get("microphone_role") != microphone_role
        or freeze.get("canonical_training_eligible") is not False
        or freeze.get("holdout_used_for_generation_or_selection") is not False
    ):
        raise ValueError(f"fit.{microphone_role}.freeze identity가 다릅니다")
    # raw_sha는 candidate JSON에 직접 없으므로 authority/provenance/operator 공통 SHA에서
    # 결속한다. 인자로 받아 lowercase authority임을 여기서도 확인한다.
    require_sha256(raw_sha, label="fit source raw SHA")
    return {
        "fit_a": fit_a,
        "fit_b": fit_b,
        "fit_a_sha": fit_a_sha,
        "fit_b_sha": fit_b_sha,
        "freeze": freeze,
        "freeze_sha": freeze_sha,
        "paths": (fit_a_path, fit_b_path, freeze_path),
        "subbands_hz": fit_a_subbands,
        "scores": (fit_a_score, fit_b_score, cross_a, cross_b),
    }


@dataclass(frozen=True)
class CausalTrainingAuthorityData:
    authority_path: Path
    authority_file_sha256: str
    authority_evidence_sha256: str
    payload: dict[str, Any]
    operator: JointCausalOperatorData
    plant_delays: PlantDelays
    timing_contract: TrainingTimingContract
    primary_history_samples: int
    secondary_history_samples: int
    referenced_paths: tuple[Path, ...]
    inline_receipt_sha256: dict[str, str]


def load_causal_training_authority(
    authority_path: str | Path,
    *,
    expected_file_sha256: str,
    expected_evidence_sha256: str,
    require_live_authority: bool = True,
) -> CausalTrainingAuthorityData:
    """v4 envelope, 모든 외부 bytes, fit/cross/holdout/stationarity를 fail-closed 검증."""

    path = Path(authority_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("causal training authority는 symlink 아닌 regular file이어야 합니다")
    file_sha = require_sha256(expected_file_sha256, label="authority file SHA")
    evidence_sha = require_sha256(
        expected_evidence_sha256, label="authority evidence SHA"
    )
    authority_bytes = path.read_bytes()
    if hashlib.sha256(authority_bytes).hexdigest() != file_sha:
        raise ValueError("causal training authority file bytes가 config와 다릅니다")
    payload = _exact(
        _json_bytes_no_duplicates(authority_bytes, label=str(path)),
        _AUTHORITY_KEYS,
        label="v4 authority",
    )
    body = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if (
        payload["schema"] != v4.TRAINING_AUTHORITY_ENVELOPE_SCHEMA
        or payload["authority"] != v4.TRAINING_AUTHORITY_SCHEMA
        or payload["status"] != "PASS"
        or payload["canonical_training_eligible"] is not True
        or payload["synthetic_fixture"] is not False
        or payload["control_band_contract_sha256"]
        != ControlBandContract.broadband_point_control().digest()
        or int(payload["sample_rate_hz"]) != 48_000
        or int(payload["block_size"]) != 256
        or payload["latency"] != "low"
        or int(payload["handoff_extra_samples"]) != 256
        or not str(payload["capture_id"]).strip()
        or payload["evidence_sha256"] != _json_sha256(body)
        or payload["evidence_sha256"] != evidence_sha
    ):
        raise ValueError("v4 causal training authority top-level 계약 위반")
    if require_live_authority:
        live = v4.LIVE_AUTHORITY
        if live is None:
            raise CausalTrainingAuthorityUnavailable(
                "v4 LIVE_AUTHORITY가 None입니다 — fixture/자체 봉인 JSON은 canonical "
                "training admission을 열 수 없습니다"
            )
        if live != v4.TRAINING_AUTHORITY_SCHEMA:
            raise ValueError("reviewed v4 LIVE_AUTHORITY schema가 현재 authority와 다릅니다")
        # v4 diagnostic publisher에는 clock receipt와 change-point receipt의 canonical
        # exact nested schema가 없다. self-sealed ``passed=true``만으로 training 권위를
        # 열면 raw lineage가 없는 JSON도 통과하므로, 스키마가 별도 review로 고정될
        # 때까지 LIVE_AUTHORITY 문자열이 생겨도 admission을 열지 않는다.
        if not EXACT_CLOCK_CHANGE_POINT_VALIDATOR_AVAILABLE:
            raise CausalTrainingAuthorityUnavailable(
                "BLOCKED_MISSING_EXACT_CLOCK_CHANGE_POINT_VALIDATOR: v4 diagnostic "
                "clock/change-point receipt는 canonical training evidence가 아닙니다"
            )
    authority_dir = path.parent

    operator_ref = _exact(
        payload["operator"], _OPERATOR_REFERENCE_KEYS, label="authority.operator"
    )
    if operator_ref["schema"] != "fullband_causal_joint_fir_operator_reference_v4":
        raise ValueError("authority operator reference schema가 다릅니다")
    operator_path = _relative_regular_file(
        authority_dir, operator_ref["npz_path"], label="authority operator NPZ"
    )
    operator = load_joint_causal_operator_npz(
        operator_path,
        expected_file_sha256=operator_ref["npz_file_sha256"],
        expected_internal_sha256=operator_ref["npz_internal_sha256"],
    )
    if (
        operator.primary_fir_sha256 != operator_ref["primary_fir_sha256"]
        or operator.secondary_fir_sha256 != operator_ref["secondary_fir_sha256"]
        or operator.source_submitted_pcm_sha256
        != operator_ref["source_submitted_pcm_sha256"]
        or operator.source_raw_sha256 != operator_ref["source_raw_sha256"]
        or operator.fit_freeze_sha256 != operator_ref["fit_freeze_sha256"]
        or int(operator_ref["support_samples"]) != operator.support_samples
        or list(map(int, operator_ref["coarse_delay_samples"]))
        != [operator.primary_coarse_delay_samples, operator.secondary_coarse_delay_samples]
        or list(map(float, operator_ref["fractional_delay_samples"]))
        != [operator.primary_fractional_delay_samples, operator.secondary_fractional_delay_samples]
        or list(map(int, operator_ref["post_onset_peak_index"]))
        != [
            int(np.argmax(np.abs(operator.primary_post_onset_fir))),
            int(np.argmax(np.abs(operator.secondary_post_onset_fir))),
        ]
    ):
        raise ValueError("authority operator metadata/FIR bytes가 joint NPZ와 다릅니다")

    plant_delays = PlantDelays.model_validate(operator_ref["plant_delays_payload"])
    if (
        _json_sha256(plant_delays.model_dump()) != operator_ref["plant_delays_sha256"]
        or plant_delays.primary_delay_samples != operator.primary_coarse_delay_samples
        or plant_delays.secondary_delay_samples != operator.secondary_coarse_delay_samples
        or plant_delays.handoff_samples != int(payload["handoff_extra_samples"])
        or plant_delays.sample_rate != operator.sample_rate_hz
    ):
        raise ValueError("authority PlantDelays payload/SHA가 joint NPZ와 다릅니다")
    timing = TrainingTimingContract.derive(
        primary_fir=operator.primary_post_onset_fir,
        plant_delays=plant_delays,
    )
    expected_effective = [
        timing.primary_effective_delay_samples,
        operator.secondary_coarse_delay_samples
        + int(np.argmax(np.abs(operator.secondary_post_onset_fir))),
    ]
    if list(map(int, operator_ref["effective_delay_samples"])) != expected_effective:
        raise ValueError("authority P/S effective delay가 FIR peak에서 유도되지 않았습니다")
    bulk = list(map(float, operator_ref["bulk_delay_samples_fractional"]))
    expected_bulk = [
        operator.primary_coarse_delay_samples + operator.primary_fractional_delay_samples,
        operator.secondary_coarse_delay_samples + operator.secondary_fractional_delay_samples,
    ]
    if len(bulk) != 2 or any(abs(a - b) > 1.0e-12 for a, b in zip(bulk, expected_bulk)):
        raise ValueError("authority fractional bulk delay가 coarse+residual과 다릅니다")

    provenance = _exact(
        payload["provenance"], _PROVENANCE_KEYS, label="authority.provenance"
    )
    commit = str(provenance["repository_commit"])
    if (
        len(commit) != 40
        or any(character not in _HEX for character in commit)
        or provenance["repository_dirty"] is not False
        or int(provenance["xrun_count"]) != 0
        or int(provenance["clip_count"]) != 0
    ):
        raise ValueError("authority source commit/dirty/xrun/clip 계약 위반")
    referenced: list[Path] = [path, operator_path]
    external: dict[str, Path] = {}
    external_bytes: dict[str, bytes] = {}
    for role in ("signal_plan", "submitted_pcm", "raw", "analysis"):
        external_path = _relative_regular_file(
            authority_dir, provenance[f"{role}_path"], label=f"provenance.{role}"
        )
        immutable_bytes = external_path.read_bytes()
        if hashlib.sha256(immutable_bytes).hexdigest() != require_sha256(
            provenance[f"{role}_file_sha256"], label=f"{role} file SHA"
        ):
            raise ValueError(f"provenance {role} file bytes가 authority와 다릅니다")
        external[role] = external_path
        external_bytes[role] = immutable_bytes
        referenced.append(external_path)
    plan = _json_bytes_no_duplicates(
        external_bytes["signal_plan"], label=str(external["signal_plan"])
    )
    claimed_plan_sha = require_sha256(
        plan.get("canonical_payload_sha256"), label="signal plan payload SHA"
    )
    plan_body = {
        key: value for key, value in plan.items() if key != "canonical_payload_sha256"
    }
    if (
        _json_sha256(plan_body) != claimed_plan_sha
        or claimed_plan_sha != provenance["signal_plan_payload_sha256"]
    ):
        raise ValueError("signal plan canonical payload SHA가 다릅니다")
    submitted_pcm = _load_submitted_pcm(
        external["submitted_pcm"], file_bytes=external_bytes["submitted_pcm"]
    )
    submitted_array_sha = hashlib.sha256(
        submitted_pcm.tobytes(order="C")
    ).hexdigest()
    plan_output = plan.get("output")
    if (
        not isinstance(plan_output, dict)
        or int(plan_output.get("frames", -1)) != int(submitted_pcm.shape[0])
        or int(plan_output.get("peak_pcm", -1))
        != int(np.max(np.abs(submitted_pcm.astype(np.int32))))
        or plan_output.get("pcm_sha256") != submitted_array_sha
        or operator.source_submitted_pcm_sha256 != submitted_array_sha
    ):
        raise ValueError("signal plan/operator/submitted int16 PCM lineage가 다릅니다")
    # source_raw_sha256는 raw NPZ 원본 file bytes의 SHA다. internal array
    # digest와 별도로 둘 다 맞아야 같은 capture generation이다.
    if provenance["raw_file_sha256"] != operator.source_raw_sha256:
        raise ValueError("operator/provenance raw file lineage가 다릅니다")
    analysis = _json_bytes_no_duplicates(
        external_bytes["analysis"], label=str(external["analysis"])
    )
    if _json_sha256(analysis) != provenance["analysis_internal_sha256"]:
        raise ValueError("analysis canonical internal SHA가 다릅니다")
    try:
        with np.load(io.BytesIO(external_bytes["raw"]), allow_pickle=False) as archive:
            raw_arrays = {
                key: np.array(archive[key], copy=True, order="C")
                for key in archive.files
            }
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("authority raw NPZ를 안전하게 읽을 수 없습니다") from exc
    if _archive_digest(
        raw_arrays, domain="fullband_causal_v4_raw_array_archive_v1"
    ) != provenance["raw_internal_sha256"]:
        raise ValueError("raw NPZ internal array digest가 다릅니다")
    if _archive_digest(
        raw_arrays,
        domain="fullband_causal_v4_callback_arrays_v1",
        selected=_CALLBACK_ARRAY_KEYS,
    ) != provenance["callback_arrays_sha256"]:
        raise ValueError("raw callback arrays digest가 다릅니다")
    for name in (
        "analysis_code_sha256", "environment_receipt_sha256",
        "level_evidence_sha256", "hardware_fingerprint_sha256",
    ):
        require_sha256(provenance[name], label=f"provenance.{name}")

    fit = _exact(payload["fit"], {"passed", "err", "ref", "receipt_sha256"}, label="fit")
    if fit["passed"] is not True or fit["receipt_sha256"] != _json_sha256(
        {key: value for key, value in fit.items() if key != "receipt_sha256"}
    ):
        raise ValueError("fit aggregate receipt가 PASS/sealed가 아닙니다")
    fit_roles = {
        microphone: _validate_fit_role(
            fit[microphone], authority_dir=authority_dir,
            microphone_role=microphone,
            submitted_sha=operator.source_submitted_pcm_sha256,
            raw_sha=operator.source_raw_sha256,
            contract_sha256=str(payload["control_band_contract_sha256"]),
        )
        for microphone in ("err", "ref")
    }
    for role_data in fit_roles.values():
        referenced.extend(role_data["paths"])
    err_freeze = fit_roles["err"]["freeze"]
    primary_frozen = np.ascontiguousarray(
        np.asarray(err_freeze.get("primary_post_onset_fir"), dtype="<f8")
    )
    secondary_frozen = np.ascontiguousarray(
        np.asarray(err_freeze.get("secondary_post_onset_fir"), dtype="<f8")
    )
    if (
        fit_roles["err"]["freeze_sha"] != operator.fit_freeze_sha256
        or not np.array_equal(primary_frozen, operator.primary_post_onset_fir)
        or not np.array_equal(secondary_frozen, operator.secondary_post_onset_fir)
    ):
        raise ValueError("ERR frozen fit와 joint operator FIR bytes가 다릅니다")

    holdout = _exact(
        payload["holdout"], {"passed", "err", "ref", "receipt_sha256"}, label="holdout"
    )
    if holdout["passed"] is not True or holdout["receipt_sha256"] != _json_sha256(
        {key: value for key, value in holdout.items() if key != "receipt_sha256"}
    ):
        raise ValueError("holdout aggregate receipt가 PASS/sealed가 아닙니다")
    for microphone in ("err", "ref"):
        role = _exact(
            holdout[microphone], {"holdout_score", "terminal_receipt"},
            label=f"holdout.{microphone}",
        )
        score = _validate_score(
            role["holdout_score"], candidate=fit_roles[microphone]["freeze"],
            target_role="holdout", microphone_role=microphone,
            contract_sha256=str(payload["control_band_contract_sha256"]),
            label=f"holdout.{microphone}.score",
        )
        terminal = _sealed_receipt(
            role["terminal_receipt"], schema="terminal_holdout_validation_v4",
            label=f"holdout.{microphone}.terminal",
        )
        if (
            terminal.get("freeze_sha256") != fit_roles[microphone]["freeze_sha"]
            or terminal.get("holdout_score_receipt_sha256") != score["receipt_sha256"]
            or terminal.get("subband_rows") != score["subband_rows"]
            or terminal.get("all_subbands_passed") is not True
            or terminal.get("passed") is not True
            or terminal.get("support_reselection_after_holdout_forbidden") is not True
        ):
            raise ValueError(f"holdout.{microphone} terminal receipt가 score와 다릅니다")

    stationarity = _exact(
        payload["stationarity"],
        {"passed", "err", "ref", "change_point_receipt", "change_point_receipt_sha256", "receipt_sha256"},
        label="stationarity",
    )
    if stationarity["passed"] is not True or stationarity["receipt_sha256"] != _json_sha256(
        {key: value for key, value in stationarity.items() if key != "receipt_sha256"}
    ):
        raise ValueError("stationarity aggregate receipt가 PASS/sealed가 아닙니다")
    stationarity_receipts = {
        microphone: _validate_stationarity(
            stationarity[microphone],
            fit_a_sha=fit_roles[microphone]["fit_a_sha"],
            fit_b_sha=fit_roles[microphone]["fit_b_sha"],
            contract_sha256=str(payload["control_band_contract_sha256"]),
            subbands_hz=fit_roles[microphone]["subbands_hz"],
            label=f"stationarity.{microphone}",
        )
        for microphone in ("err", "ref")
    }
    change_point = _sealed_receipt(
        stationarity["change_point_receipt"], schema=None,
        label="stationarity.change_point",
    )
    if (
        change_point.get("passed") is not True
        or change_point["receipt_sha256"] != stationarity["change_point_receipt_sha256"]
    ):
        raise ValueError("clock/FIR change-point receipt가 FAIL/불일치입니다")

    clock = _exact(
        payload["clock"],
        {"passed", "receipt", "receipt_sha256", "clock_witness_kind", "independent_electrical_witness_present"},
        label="clock",
    )
    clock_receipt = _sealed_receipt(clock["receipt"], schema=None, label="clock.receipt")
    if (
        clock["passed"] is not True
        or clock_receipt.get("passed") is not True
        or clock_receipt["receipt_sha256"] != clock["receipt_sha256"]
        or clock["clock_witness_kind"]
        != "continuous_acoustic_reserved_pilot_v4"
        or not isinstance(clock["independent_electrical_witness_present"], bool)
    ):
        raise ValueError("continuous clock witness receipt가 FAIL/불일치입니다")

    inline = {
        "clock": clock["receipt_sha256"],
        "fit": fit["receipt_sha256"],
        "holdout": holdout["receipt_sha256"],
        "stationarity": stationarity["receipt_sha256"],
        "change_point": stationarity["change_point_receipt_sha256"],
        "err_fit_stationarity": stationarity_receipts["err"]["receipt_sha256"],
        "ref_fit_stationarity": stationarity_receipts["ref"]["receipt_sha256"],
    }
    primary_history = (
        operator.primary_coarse_delay_samples + operator.support_samples
    )
    secondary_history = (
        operator.secondary_coarse_delay_samples
        + int(payload["handoff_extra_samples"])
        + operator.support_samples
    )
    return CausalTrainingAuthorityData(
        authority_path=path,
        authority_file_sha256=file_sha,
        authority_evidence_sha256=evidence_sha,
        payload=payload,
        operator=operator,
        plant_delays=plant_delays,
        timing_contract=timing,
        primary_history_samples=primary_history,
        secondary_history_samples=secondary_history,
        referenced_paths=tuple(referenced),
        inline_receipt_sha256=inline,
    )


__all__ = [
    "CausalTrainingAuthorityData",
    "CausalTrainingAuthorityUnavailable",
    "EXACT_CLOCK_CHANGE_POINT_VALIDATOR_AVAILABLE",
    "JointCausalOperatorData",
    "NPZ_INTERNAL_DIGEST_DOMAIN",
    "load_joint_causal_operator_npz",
    "load_causal_training_authority",
    "operator_npz_internal_sha256",
    "repository_regular_file",
    "require_sha256",
    "sha256_file",
]
