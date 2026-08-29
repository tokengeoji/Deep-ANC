"""광대역 recorded v3 coverage receipt의 fail-closed 검증.

source에 고역이 있다는 사실만으로는 학습 coverage가 아니다. 이 receipt는 실제 ERR target
``d``의 segment별 coherence와 energy density, native Nyquist, sub-sample alignment,
lineage group을 manifest/P/S/timing SHA와 함께 봉인한다. 결과 평균이 한 family 또는 한
subband 실패를 숨길 수 없도록 split×family×band를 독립 재집계한다.

이 모듈은 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..dsp.control_band_contract import (
    ControlBandContract,
    max_timing_error_samples_for_attenuation,
)


BROADBAND_COVERAGE_RECEIPT_SCHEMA = "recorded_broadband_coverage_receipt_v3"
BROADBAND_SOURCE_TRANSFORM_SCHEMA = "broadband_source_transform_receipt_v1"
NATIVE_EXACT_TARGET_RATE_ROLE = "native_exact_48k"
NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE = (
    "native_above_target_nyquist_resampled_once"
)
MIN_SOURCE_ERR_COHERENCE = 0.60
MIN_TARGET_D_DENSITY_RATIO = 0.25
MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND = 4
# 한 transient 하나로 group을 채우는 것을 막는 provisional hard floor. pilot 이전에 고정하며
# 결과를 본 뒤 낮추지 않는다. 더 강한 campaign 값은 policy에 기록할 수 있다.
MIN_JOINT_SEGMENTS_PER_GROUP = 8
MIN_JOINT_SEGMENT_FRACTION_PER_GROUP = 0.50
REQUIRED_SPLITS = ("train", "val", "test")
_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return text


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minimum_native_sample_rate_hz(contract: ControlBandContract) -> int:
    """요구 상단을 native Nyquist가 실제로 포함하기 위한 정수 sample-rate 하한."""

    if contract.role != "broadband_point_control":
        raise ValueError("native 광대역 source 하한에는 broadband contract가 필요합니다")
    return int(math.ceil(2.0 * float(contract.required_excitation_upper_hz)))


def build_broadband_coverage_policy(
    *,
    minimum_joint_segments_per_group: int = MIN_JOINT_SEGMENTS_PER_GROUP,
    minimum_joint_segment_fraction_per_group: float = (
        MIN_JOINT_SEGMENT_FRACTION_PER_GROUP
    ),
) -> dict[str, Any]:
    segments = int(minimum_joint_segments_per_group)
    fraction = float(minimum_joint_segment_fraction_per_group)
    if segments < MIN_JOINT_SEGMENTS_PER_GROUP:
        raise ValueError("group별 joint segment 하한을 8보다 낮출 수 없습니다")
    if not math.isfinite(fraction) or not (
        MIN_JOINT_SEGMENT_FRACTION_PER_GROUP <= fraction <= 1.0
    ):
        raise ValueError("group별 joint segment fraction을 0.50보다 낮출 수 없습니다")
    policy = {
        "minimum_source_err_coherence": MIN_SOURCE_ERR_COHERENCE,
        "minimum_target_d_density_ratio": MIN_TARGET_D_DENSITY_RATIO,
        "minimum_independent_groups_per_split_family_band": (
            MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND
        ),
        "minimum_joint_segments_per_group": segments,
        "minimum_joint_segment_fraction_per_group": fraction,
        "required_splits": list(REQUIRED_SPLITS),
        "native_nyquist_required": True,
        "upsampled_source_cannot_count_as_native_coverage": True,
        "native_bandwidth_must_preexist_before_rate_conversion": True,
        "processed_48k_is_not_native_rate_evidence": True,
        "non_48k_source_requires_one_polyphase_transform_receipt": True,
        "subsample_alignment_and_clock_witness_required": True,
    }
    policy["policy_sha256"] = hashlib.sha256(_canonical_json(policy)).hexdigest()
    return policy


def seal_broadband_coverage_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("evidence_sha256", None)
    sealed["evidence_sha256"] = hashlib.sha256(_canonical_json(sealed)).hexdigest()
    return sealed


def _regular_file_reference(
    value: object,
    *,
    root: Path,
    label: str,
    require_local_files: bool,
) -> dict[str, Any]:
    reference = _exact_keys(value, {"path", "size_bytes", "sha256"}, label=label)
    path_text = str(reference["path"])
    if not path_text or Path(path_text).is_absolute():
        raise ValueError(f"{label}.path는 저장소 상대경로여야 합니다")
    candidate = Path(os.path.abspath(root / path_text))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}.path가 저장소 밖입니다") from exc
    if any((root.joinpath(*Path(path_text).parts[:index])).is_symlink() for index in range(1, len(Path(path_text).parts) + 1)):
        raise ValueError(f"{label}.path에 symlink가 있습니다")
    size = reference["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_bytes가 양의 정수가 아닙니다")
    digest = _sha(reference["sha256"], label=f"{label}.sha256")
    if require_local_files:
        if not candidate.is_file():
            raise ValueError(f"{label} 파일이 없습니다: {candidate}")
        stat = candidate.stat()
        if stat.st_size != size or sha256_file(candidate) != digest:
            raise ValueError(f"{label} bytes가 receipt와 다릅니다")
    return {"path": path_text, "size_bytes": size, "sha256": digest}


def _canonical_payload_file_reference(
    value: object,
    *,
    payload: Mapping[str, Any],
    root: Path,
    label: str,
    require_local_files: bool,
) -> dict[str, Any]:
    """외부 JSON receipt가 embedded payload의 canonical bytes인지 검증한다."""

    reference = _regular_file_reference(
        value,
        root=root,
        label=label,
        require_local_files=require_local_files,
    )
    encoded = _canonical_json(payload)
    expected_sha = hashlib.sha256(encoded).hexdigest()
    if reference["size_bytes"] != len(encoded) or reference["sha256"] != expected_sha:
        raise ValueError(f"{label}가 embedded canonical payload와 다릅니다")
    return reference


def _audio_header(
    reference: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> tuple[int, str, str]:
    """검증된 로컬 reference의 실제 lossless audio header를 읽는다."""

    import soundfile as sf

    path = root / str(reference["path"])
    try:
        info = sf.info(str(path))
    except RuntimeError as exc:
        raise ValueError(f"{label} audio header를 읽을 수 없습니다") from exc
    subtype = str(info.subtype or "")
    if not (
        subtype.startswith("PCM_") or subtype in {"FLOAT", "DOUBLE"}
    ):
        raise ValueError(f"{label}는 lossless PCM/FLAC audio가 아닙니다: {subtype}")
    return int(info.samplerate), str(info.format or ""), subtype


def _validate_source_transform(
    source: object,
    *,
    contract: ControlBandContract,
    root: Path,
    session_id: str,
    require_local_files: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """native raw와 48 kHz 재생본, 그 사이 변환을 독립적으로 결속한다."""

    value = _exact_keys(
        source,
        {
            "raw_native_file",
            "processed_file",
            "native_sample_rate",
            "native_nyquist_hz",
            "transform_receipt",
        },
        label=f"{session_id}.source",
    )
    raw_native = _regular_file_reference(
        value["raw_native_file"],
        root=root,
        label=f"{session_id}.source.raw_native_file",
        require_local_files=require_local_files,
    )
    processed = _regular_file_reference(
        value["processed_file"],
        root=root,
        label=f"{session_id}.source.processed_file",
        require_local_files=require_local_files,
    )
    native_rate_raw = value["native_sample_rate"]
    if isinstance(native_rate_raw, bool) or not isinstance(native_rate_raw, int):
        raise ValueError(f"{session_id}: native sample rate가 정수가 아닙니다")
    native_rate = int(native_rate_raw)
    native_nyquist = float(value["native_nyquist_hz"])
    if native_rate <= 0 or not math.isclose(
        native_nyquist, native_rate / 2.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(f"{session_id}: native sample rate/Nyquist가 다릅니다")
    minimum_rate = minimum_native_sample_rate_hz(contract)
    if native_rate < minimum_rate or native_nyquist < contract.required_excitation_upper_hz:
        raise ValueError(
            f"{session_id}: native Nyquist가 8k octave를 덮지 못합니다 "
            f"(fs {native_rate} < {minimum_rate})"
        )

    transform = _exact_keys(
        value["transform_receipt"],
        {"file", "payload"},
        label=f"{session_id}.source.transform_receipt",
    )
    transform_payload = _exact_keys(
        transform["payload"],
        {
            "schema",
            "processing_role",
            "raw_native_sha256",
            "processed_sha256",
            "input_sample_rate_hz",
            "output_sample_rate_hz",
            "resample_count",
            "native_bandwidth_coverage_verified",
            "synthetic_bandwidth_claimed",
            "lossless_native",
            "resampler",
        },
        label=f"{session_id}.source.transform_receipt.payload",
    )
    if transform_payload["schema"] != BROADBAND_SOURCE_TRANSFORM_SCHEMA:
        raise ValueError(f"{session_id}: source transform schema가 다릅니다")
    transform_file = _canonical_payload_file_reference(
        transform["file"],
        payload=transform_payload,
        root=root,
        label=f"{session_id}.source.transform_receipt.file",
        require_local_files=require_local_files,
    )
    if _sha(
        transform_payload["raw_native_sha256"],
        label=f"{session_id} transform raw SHA",
    ) != raw_native["sha256"] or _sha(
        transform_payload["processed_sha256"],
        label=f"{session_id} transform processed SHA",
    ) != processed["sha256"]:
        raise ValueError(f"{session_id}: source transform SHA lineage가 다릅니다")
    if (
        transform_payload["input_sample_rate_hz"] != native_rate
        or transform_payload["output_sample_rate_hz"] != contract.sample_rate
    ):
        raise ValueError(f"{session_id}: source transform sample rate lineage가 다릅니다")
    if transform_payload["native_bandwidth_coverage_verified"] is not True:
        raise ValueError(f"{session_id}: native bandwidth coverage 확인이 없습니다")
    if transform_payload["synthetic_bandwidth_claimed"] is not False:
        raise ValueError(f"{session_id}: resample로 새 bandwidth를 만들었다고 주장할 수 없습니다")
    if transform_payload["lossless_native"] is not True:
        raise ValueError(f"{session_id}: native source가 lossless가 아닙니다")

    role = str(transform_payload["processing_role"])
    count = transform_payload["resample_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{session_id}: resample_count가 정수가 아닙니다")
    resampler = transform_payload["resampler"]
    if role == NATIVE_EXACT_TARGET_RATE_ROLE:
        if native_rate != contract.sample_rate or count != 0 or resampler is not None:
            raise ValueError(f"{session_id}: native exact-48k 역할과 변환이 모순됩니다")
        if raw_native["sha256"] != processed["sha256"]:
            raise ValueError(f"{session_id}: identity source의 raw/processed bytes가 다릅니다")
    elif role == NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE:
        if native_rate == contract.sample_rate or count != 1:
            raise ValueError(f"{session_id}: native-resampled-once 역할과 rate/count가 모순됩니다")
        if raw_native["sha256"] == processed["sha256"]:
            raise ValueError(f"{session_id}: resampled source의 raw/processed bytes가 같습니다")
        resampler_row = _exact_keys(
            resampler,
            {
                "algorithm",
                "implementation",
                "version",
                "parameters_sha256",
                "verified_passband_upper_hz",
                "frequency_response_evidence",
            },
            label=f"{session_id}.source.transform_receipt.payload.resampler",
        )
        if resampler_row["algorithm"] != "polyphase_fir":
            raise ValueError(f"{session_id}: 변환은 polyphase FIR 1회여야 합니다")
        if not str(resampler_row["implementation"]).strip() or not str(
            resampler_row["version"]
        ).strip():
            raise ValueError(f"{session_id}: resampler 구현/version이 비었습니다")
        _sha(
            resampler_row["parameters_sha256"],
            label=f"{session_id} resampler parameters SHA",
        )
        passband_upper = float(resampler_row["verified_passband_upper_hz"])
        maximum_possible = min(native_nyquist, contract.sample_rate / 2.0)
        if not (
            math.isfinite(passband_upper)
            and passband_upper >= contract.required_excitation_upper_hz
            and passband_upper <= maximum_possible
        ):
            raise ValueError(f"{session_id}: resampler passband가 8k octave를 보존하지 못합니다")
        _regular_file_reference(
            resampler_row["frequency_response_evidence"],
            root=root,
            label=f"{session_id}.source.resampler.frequency_response_evidence",
            require_local_files=require_local_files,
        )
    else:
        raise ValueError(f"{session_id}: source processing 역할이 canonical 집합 밖입니다")

    if require_local_files:
        raw_header_rate, _, _ = _audio_header(
            raw_native,
            root=root,
            label=f"{session_id}.source.raw_native_file",
        )
        processed_header_rate, _, _ = _audio_header(
            processed,
            root=root,
            label=f"{session_id}.source.processed_file",
        )
        if raw_header_rate != native_rate:
            raise ValueError(f"{session_id}: native header rate가 receipt와 다릅니다")
        if processed_header_rate != contract.sample_rate:
            raise ValueError(f"{session_id}: processed source가 실제 48k가 아닙니다")
    return raw_native, processed, transform_file


def _same_band(actual: object, expected: Sequence[float]) -> bool:
    try:
        values = tuple(float(value) for value in actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return len(values) == 2 and all(
        math.isclose(left, float(right), rel_tol=0.0, abs_tol=1.0e-9)
        for left, right in zip(values, expected, strict=True)
    )


def _validate_recorded_v2_session_binding(
    reference: object,
    *,
    root: Path,
    session_id: str,
    split: str,
    family: str,
    group: str,
    lineage: str,
    raw_native_sha256: str,
    processed_sha256: str,
    transform_receipt_sha256: str,
    mics_reference: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    """campaign mode에서 raw→timewarp→persisted WAV→coverage trust chain을 재검산한다."""

    import soundfile as sf

    from .recorded_v2_capture import (
        RECORDED_V2_SESSION_SCHEMA,
        RECORDED_V2_TIMEWARP_SCHEMA,
        SOURCE_FRAMES,
        capture_contract,
        validate_raw_capture_bundle,
        validate_stored_actual_err_coverage,
        validate_timewarp_receipt,
    )

    session_ref = _regular_file_reference(
        reference,
        root=root,
        label=f"{session_id}.recorded_v2_session",
        require_local_files=True,
    )
    session_path = root / session_ref["path"]
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{session_id}: recorded-v2 session JSON 오류") from exc
    session = _exact_keys(
        session,
        {
            "schema",
            "role",
            "session_id",
            "split",
            "source_family",
            "group_id",
            "lineage_id",
            "source_plan_row_sha256",
            "native_source_sha256",
            "processed_source_sha256",
            "transform_receipt_sha256",
            "capture_contract_sha256",
            "raw_capture",
            "timewarp_receipt",
            "alignment",
            "artifacts",
            "coverage_evidence_sha256",
            "evidence_sha256",
        },
        label=f"{session_id}.recorded_v2_session payload",
    )
    if session["schema"] != RECORDED_V2_SESSION_SCHEMA or session["role"] != (
        "canonical_aligned_after_immutable_raw"
    ):
        raise ValueError(f"{session_id}: recorded-v2 session 역할/schema가 다릅니다")
    expected_digest = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in session.items() if key != "evidence_sha256"}
        )
    ).hexdigest()
    if _sha(session["evidence_sha256"], label=f"{session_id} session evidence SHA") != expected_digest:
        raise ValueError(f"{session_id}: session evidence SHA가 재계산과 다릅니다")
    expected_identity = {
        "session_id": session_id,
        "split": split,
        "source_family": family,
        "group_id": group,
        "lineage_id": lineage,
        "native_source_sha256": raw_native_sha256,
        "processed_source_sha256": processed_sha256,
        "transform_receipt_sha256": transform_receipt_sha256,
    }
    if any(session.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"{session_id}: session identity/source transform lineage가 다릅니다")
    _sha(session["source_plan_row_sha256"], label=f"{session_id} source-plan row SHA")
    if session["capture_contract_sha256"] != capture_contract()["contract_sha256"]:
        raise ValueError(f"{session_id}: capture contract SHA가 다릅니다")

    raw_ref = _regular_file_reference(
        session["raw_capture"],
        root=root,
        label=f"{session_id}.raw_capture",
        require_local_files=True,
    )
    raw_bundle = validate_raw_capture_bundle(
        (root / raw_ref["path"]).parent,
        repository_root=root,
        require_valid_for_analysis=True,
    )
    if (
        raw_bundle["receipt_file_sha256"] != raw_ref["sha256"]
        or raw_bundle["receipt"]["source_plan_row_sha256"]
        != session["source_plan_row_sha256"]
    ):
        raise ValueError(f"{session_id}: raw capture/session plan lineage가 다릅니다")
    warp_ref = _regular_file_reference(
        session["timewarp_receipt"],
        root=root,
        label=f"{session_id}.timewarp_receipt",
        require_local_files=True,
    )
    try:
        warp_payload = json.loads((root / warp_ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{session_id}: timewarp receipt JSON 오류") from exc
    if warp_payload.get("schema") != RECORDED_V2_TIMEWARP_SCHEMA:
        raise ValueError(f"{session_id}: timewarp receipt schema가 다릅니다")
    warp = validate_timewarp_receipt(
        warp_payload,
        repository_root=root,
        expected_raw_capture_sha256=raw_bundle["raw_capture_sha256"],
        expected_submitted_pcm_sha256=raw_bundle["submitted_output_pcm_sha256"],
        expected_mics_pcm_sha256=raw_bundle["mics_raw_pcm_sha256"],
    )
    alignment = _exact_keys(
        session["alignment"],
        {
            "schema",
            "method",
            "timewarp_map_sha256",
            "source_aligned_float32_sha256",
            "mics_aligned_float32_sha256",
            "source_frames",
            "adc_position_min",
            "adc_position_max",
            "evidence_sha256",
        },
        label=f"{session_id}.alignment",
    )
    alignment_digest = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in alignment.items() if key != "evidence_sha256"}
        )
    ).hexdigest()
    if (
        _sha(alignment["evidence_sha256"], label=f"{session_id} alignment SHA")
        != alignment_digest
        or alignment["timewarp_map_sha256"] != warp["map_sha256"]
        or alignment["source_frames"] != SOURCE_FRAMES
    ):
        raise ValueError(f"{session_id}: aligned arrays/timewarp evidence가 다릅니다")

    artifacts = _exact_keys(
        session["artifacts"],
        {"source.wav", "source_aligned.wav", "mics.wav", "coverage.json"},
        label=f"{session_id}.artifacts",
    )
    session_dir = session_path.parent

    def artifact(name: str) -> tuple[Path, dict[str, Any]]:
        row = _exact_keys(
            artifacts[name], {"path", "size_bytes", "sha256"}, label=f"{session_id}.{name}"
        )
        if row["path"] != name:
            raise ValueError(f"{session_id}: session artifact basename이 다릅니다")
        path = session_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{session_id}: session artifact가 없습니다: {name}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != _sha(
            row["sha256"], label=f"{session_id}.{name} SHA"
        ):
            raise ValueError(f"{session_id}: session artifact bytes가 다릅니다: {name}")
        return path, row

    source_path, _ = artifact("source_aligned.wav")
    mics_path, mics_artifact = artifact("mics.wav")
    artifact("source.wav")
    coverage_path, _ = artifact("coverage.json")
    if (
        mics_reference["path"] != mics_path.relative_to(root).as_posix()
        or mics_reference["sha256"] != mics_artifact["sha256"]
        or mics_reference["size_bytes"] != mics_artifact["size_bytes"]
    ):
        raise ValueError(f"{session_id}: coverage mics ref와 session mics artifact가 다릅니다")
    source, source_rate = sf.read(source_path, dtype="float32", always_2d=False)
    microphones, mics_rate = sf.read(mics_path, dtype="float32", always_2d=True)
    if source_rate != 48_000 or mics_rate != 48_000:
        raise ValueError(f"{session_id}: persisted WAV sample rate가 48kHz가 아닙니다")
    if hashlib.sha256(np.asarray(source, dtype=np.float32).tobytes()).hexdigest() != alignment[
        "source_aligned_float32_sha256"
    ] or hashlib.sha256(np.asarray(microphones, dtype=np.float32).tobytes()).hexdigest() != alignment[
        "mics_aligned_float32_sha256"
    ]:
        raise ValueError(f"{session_id}: persisted aligned array SHA가 alignment receipt와 다릅니다")
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{session_id}: actual ERR coverage JSON 오류") from exc
    coverage = _exact_keys(
        coverage,
        {
            "schema",
            "role",
            "control_band_contract_sha256",
            "segments",
            "summary",
            "evidence_sha256",
        },
        label=f"{session_id}.actual ERR coverage",
    )
    coverage_digest = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in coverage.items() if key != "evidence_sha256"}
        )
    ).hexdigest()
    if (
        coverage["schema"] != "recorded_broadband_v2_actual_err_coverage_v1"
        or coverage["role"]
        != "recomputed_from_persisted_source_aligned_and_mics_err_ch0"
        or coverage["control_band_contract_sha256"]
        != ControlBandContract.broadband_point_control().digest()
        or _sha(coverage["evidence_sha256"], label=f"{session_id} coverage SHA")
        != coverage_digest
        or coverage["evidence_sha256"] != session["coverage_evidence_sha256"]
    ):
        raise ValueError(f"{session_id}: actual ERR coverage provenance가 다릅니다")
    summary = validate_stored_actual_err_coverage(
        coverage["segments"], source_aligned=source, mics=microphones
    )
    if summary != coverage["summary"] or summary["status"] != "PASS":
        raise ValueError(f"{session_id}: actual ERR coverage WAV 재계산이 PASS가 아닙니다")
    per_band: list[list[dict[str, Any]]] = [
        [] for _ in ControlBandContract.broadband_point_control().point_control_subbands_hz
    ]
    for segment in coverage["segments"]:
        for band_index in range(len(per_band)):
            per_band[band_index].append(
                {
                    "start_frame": int(segment["start_frame"]),
                    "n_frames": int(segment["n_frames"]),
                    "coherence": float(segment["coherence"][band_index]),
                    "target_density_ratio": float(
                        segment["target_density_ratio"][band_index]
                    ),
                }
            )
    return per_band


def validate_broadband_coverage_receipt(
    payload: object,
    *,
    contract: ControlBandContract,
    repository_root: str | Path,
    require_local_files: bool = True,
) -> dict[str, Any]:
    """receipt bytes·파일·lineage·coverage 집계를 모두 다시 확인한다."""

    if contract.role != "broadband_point_control":
        raise ValueError("광대역 coverage receipt에는 broadband contract가 필요합니다")
    root = Path(os.path.abspath(Path(repository_root).expanduser()))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root가 유효한 실제 directory가 아닙니다")
    receipt = _exact_keys(
        payload,
        {
            "schema",
            "role",
            "control_band_contract_sha256",
            "policy",
            "manifest",
            "plant_evidence",
            "training_timing_contract_sha256",
            "alignment_policy_sha256",
            "coverage_algorithm_sha256",
            "sessions",
            "summary",
            "evidence_sha256",
        },
        label="broadband coverage receipt",
    )
    if receipt["schema"] != BROADBAND_COVERAGE_RECEIPT_SCHEMA:
        raise ValueError("broadband coverage receipt schema가 다릅니다")
    if receipt["role"] != "campaign_readiness_not_diagnostic":
        raise ValueError("diagnostic report를 campaign receipt로 승격할 수 없습니다")
    if receipt["control_band_contract_sha256"] != contract.digest():
        raise ValueError("control-band contract SHA가 다릅니다")
    expected_digest = hashlib.sha256(
        _canonical_json({key: value for key, value in receipt.items() if key != "evidence_sha256"})
    ).hexdigest()
    if _sha(receipt["evidence_sha256"], label="receipt evidence_sha256") != expected_digest:
        raise ValueError("coverage receipt evidence SHA가 다릅니다")

    policy = dict(receipt["policy"])
    expected_policy = build_broadband_coverage_policy(
        minimum_joint_segments_per_group=int(
            policy.get("minimum_joint_segments_per_group", -1)
        ),
        minimum_joint_segment_fraction_per_group=float(
            policy.get("minimum_joint_segment_fraction_per_group", float("nan"))
        ),
    )
    if policy != expected_policy:
        raise ValueError("coverage policy가 canonical hard floor와 다릅니다")
    manifest = _regular_file_reference(
        receipt["manifest"],
        root=root,
        label="coverage manifest",
        require_local_files=require_local_files,
    )
    plant = _regular_file_reference(
        receipt["plant_evidence"],
        root=root,
        label="broadband plant evidence",
        require_local_files=require_local_files,
    )
    timing_sha = _sha(
        receipt["training_timing_contract_sha256"], label="training timing SHA"
    )
    alignment_policy_sha = _sha(
        receipt["alignment_policy_sha256"], label="alignment policy SHA"
    )
    coverage_algorithm_sha = _sha(
        receipt["coverage_algorithm_sha256"], label="coverage algorithm SHA"
    )

    sessions = receipt["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("coverage session이 비었습니다")
    seen_sessions: set[str] = set()
    group_assignment: dict[str, tuple[str, str]] = {}
    lineage_to_group: dict[str, str] = {}
    native_source_sha_to_group: dict[str, str] = {}
    processed_source_sha_to_group: dict[str, str] = {}
    seen_mics_sha: set[str] = set()
    group_band_counts: dict[tuple[str, str, str, int], list[int]] = {}
    for index, raw in enumerate(sessions):
        session_keys = {
            "session_id",
            "split",
            "source_family",
            "group_id",
            "lineage_id",
            "source",
            "mics",
            "alignment",
            "bands",
        }
        if require_local_files:
            session_keys.add("recorded_v2_session")
        session = _exact_keys(
            raw,
            session_keys,
            label=f"coverage session #{index}",
        )
        session_id = str(session["session_id"]).strip()
        split = str(session["split"])
        family = str(session["source_family"])
        group = str(session["group_id"]).strip()
        lineage = str(session["lineage_id"]).strip()
        if not session_id or session_id in seen_sessions or not group or not lineage:
            raise ValueError("session_id/group_id/lineage_id가 비었거나 중복입니다")
        seen_sessions.add(session_id)
        if split not in REQUIRED_SPLITS or family not in contract.source_families:
            raise ValueError(f"{session_id}: split/family가 canonical 집합 밖입니다")
        assignment = (split, family)
        if group in group_assignment and group_assignment[group] != assignment:
            raise ValueError(f"lineage group이 split/family를 넘나듭니다: {group}")
        group_assignment[group] = assignment
        if lineage in lineage_to_group and lineage_to_group[lineage] != group:
            raise ValueError(f"같은 lineage가 여러 group으로 갈라졌습니다: {lineage}")
        lineage_to_group[lineage] = group

        raw_native_file, processed_file, transform_file = _validate_source_transform(
            session["source"],
            contract=contract,
            root=root,
            session_id=session_id,
            require_local_files=require_local_files,
        )
        native_sha = raw_native_file["sha256"]
        if (
            native_sha in native_source_sha_to_group
            and native_source_sha_to_group[native_sha] != group
        ):
            raise ValueError("같은 native source SHA가 여러 독립 group으로 집계됐습니다")
        native_source_sha_to_group[native_sha] = group
        processed_sha = processed_file["sha256"]
        if (
            processed_sha in processed_source_sha_to_group
            and processed_source_sha_to_group[processed_sha] != group
        ):
            raise ValueError("같은 processed source SHA가 여러 독립 group으로 집계됐습니다")
        processed_source_sha_to_group[processed_sha] = group
        mics = _exact_keys(
            session["mics"], {"file", "sample_rate"}, label=f"{session_id}.mics"
        )
        mics_file = _regular_file_reference(
            mics["file"],
            root=root,
            label=f"{session_id}.mics.file",
            require_local_files=require_local_files,
        )
        if mics_file["sha256"] in seen_mics_sha:
            raise ValueError("같은 mics WAV가 여러 session으로 중복 집계됐습니다")
        seen_mics_sha.add(mics_file["sha256"])
        if int(mics["sample_rate"]) != contract.sample_rate:
            raise ValueError(f"{session_id}: mics sample rate가 48k가 아닙니다")
        alignment = _exact_keys(
            session["alignment"],
            {
                "receipt_sha256",
                "method",
                "subsample_alignment",
                "clock_witness",
                "timing_jitter_samples_by_subband",
            },
            label=f"{session_id}.alignment",
        )
        _sha(alignment["receipt_sha256"], label=f"{session_id} alignment receipt")
        if alignment["method"] not in {
            "electrical_playback_loopback",
            "pilot_fractional_warp",
        }:
            raise ValueError(f"{session_id}: broadband alignment 방법이 불명확합니다")
        if alignment["subsample_alignment"] is not True or alignment["clock_witness"] is not True:
            raise ValueError(f"{session_id}: sub-sample alignment/clock 증거가 없습니다")
        jitter = alignment["timing_jitter_samples_by_subband"]
        if not isinstance(jitter, list) or len(jitter) != len(contract.point_control_subbands_hz):
            raise ValueError(f"{session_id}: alignment jitter vector 길이가 다릅니다")
        for band, actual in zip(contract.point_control_subbands_hz, jitter, strict=True):
            value = float(actual)
            limit = max_timing_error_samples_for_attenuation(
                contract.measurement_resolution_attenuation_db,
                band[1],
                contract.sample_rate,
            )
            if not math.isfinite(value) or value < 0.0 or value > limit:
                raise ValueError(
                    f"{session_id}: {band} timing jitter가 "
                    f"{contract.measurement_resolution_attenuation_db:g}dB 예산을 넘습니다"
                )

        recorded_v2_segments_by_band: list[list[dict[str, Any]]] | None = None
        if require_local_files:
            recorded_v2_segments_by_band = _validate_recorded_v2_session_binding(
                session["recorded_v2_session"],
                root=root,
                session_id=session_id,
                split=split,
                family=family,
                group=group,
                lineage=lineage,
                raw_native_sha256=raw_native_file["sha256"],
                processed_sha256=processed_file["sha256"],
                transform_receipt_sha256=transform_file["sha256"],
                mics_reference=mics_file,
            )

        bands = session["bands"]
        if not isinstance(bands, list) or len(bands) != len(contract.point_control_subbands_hz):
            raise ValueError(f"{session_id}: coverage band row 수가 다릅니다")
        for band_index, (row_raw, expected_band) in enumerate(
            zip(bands, contract.point_control_subbands_hz, strict=True)
        ):
            row = _exact_keys(
                row_raw,
                {
                    "band_hz",
                    "n_segments",
                    "coherence_pass_segments",
                    "target_density_pass_segments",
                    "joint_pass_segments",
                    "median_coherence",
                    "median_target_density_ratio",
                    "segments",
                },
                label=f"{session_id}.band#{band_index}",
            )
            if not _same_band(row["band_hz"], expected_band):
                raise ValueError(f"{session_id}: coverage band 순서/경계가 다릅니다")
            segment_rows = row["segments"]
            if not isinstance(segment_rows, list) or not segment_rows:
                raise ValueError(f"{session_id}: band segment evidence가 비었습니다")
            if recorded_v2_segments_by_band is not None:
                expected_segments = recorded_v2_segments_by_band[band_index]
                if len(segment_rows) != len(expected_segments):
                    raise ValueError(
                        f"{session_id}: coverage segment 수가 persisted actual-ERR evidence와 다릅니다"
                    )
                for actual_segment, expected_segment in zip(
                    segment_rows, expected_segments, strict=True
                ):
                    if set(actual_segment) != set(expected_segment):
                        raise ValueError(
                            f"{session_id}: coverage segment schema가 persisted evidence와 다릅니다"
                        )
                    for key in ("start_frame", "n_frames"):
                        if actual_segment[key] != expected_segment[key]:
                            raise ValueError(
                                f"{session_id}: coverage segment population이 persisted evidence와 다릅니다"
                            )
                    for key in ("coherence", "target_density_ratio"):
                        if not math.isclose(
                            float(actual_segment[key]),
                            float(expected_segment[key]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        ):
                            raise ValueError(
                                f"{session_id}: coverage metric이 actual ERR WAV 재계산과 다릅니다"
                            )
            coherence_values: list[float] = []
            density_values: list[float] = []
            seen_ranges: set[tuple[int, int]] = set()
            for segment_index, segment_raw in enumerate(segment_rows):
                segment = _exact_keys(
                    segment_raw,
                    {"start_frame", "n_frames", "coherence", "target_density_ratio"},
                    label=f"{session_id}.band#{band_index}.segment#{segment_index}",
                )
                start_frame = segment["start_frame"]
                n_frames = segment["n_frames"]
                if (
                    isinstance(start_frame, bool)
                    or not isinstance(start_frame, int)
                    or start_frame < 0
                    or isinstance(n_frames, bool)
                    or not isinstance(n_frames, int)
                    or n_frames <= 0
                ):
                    raise ValueError(f"{session_id}: segment frame 범위가 잘못됐습니다")
                frame_range = (int(start_frame), int(n_frames))
                if frame_range in seen_ranges:
                    raise ValueError(f"{session_id}: 같은 segment를 중복 집계했습니다")
                seen_ranges.add(frame_range)
                coherence_value = float(segment["coherence"])
                density_value = float(segment["target_density_ratio"])
                if not (
                    math.isfinite(coherence_value)
                    and 0.0 <= coherence_value <= 1.0
                    and math.isfinite(density_value)
                    and density_value >= 0.0
                ):
                    raise ValueError(f"{session_id}: segment metric이 유효하지 않습니다")
                coherence_values.append(coherence_value)
                density_values.append(density_value)
            recomputed_total = len(segment_rows)
            recomputed_coherence = sum(
                value >= MIN_SOURCE_ERR_COHERENCE for value in coherence_values
            )
            recomputed_density = sum(
                value >= MIN_TARGET_D_DENSITY_RATIO for value in density_values
            )
            recomputed_joint = sum(
                coherence_value >= MIN_SOURCE_ERR_COHERENCE
                and density_value >= MIN_TARGET_D_DENSITY_RATIO
                for coherence_value, density_value in zip(
                    coherence_values, density_values, strict=True
                )
            )
            counts = [
                row["n_segments"],
                row["coherence_pass_segments"],
                row["target_density_pass_segments"],
                row["joint_pass_segments"],
            ]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
                raise ValueError(f"{session_id}: segment count가 정수가 아닙니다")
            total, coherence_count, density_count, joint_count = (int(value) for value in counts)
            if not (total > 0 and 0 <= joint_count <= min(coherence_count, density_count) <= total):
                raise ValueError(f"{session_id}: segment count 관계가 잘못됐습니다")
            if (total, coherence_count, density_count, joint_count) != (
                recomputed_total,
                recomputed_coherence,
                recomputed_density,
                recomputed_joint,
            ):
                raise ValueError(f"{session_id}: segment count가 raw metric row 재집계와 다릅니다")
            median_coherence = float(row["median_coherence"])
            median_density = float(row["median_target_density_ratio"])
            if not (
                math.isfinite(median_coherence)
                and math.isfinite(median_density)
                and 0.0 <= median_coherence <= 1.0
                and median_density >= 0.0
            ):
                raise ValueError(f"{session_id}: coverage median이 유효하지 않습니다")
            sorted_coherence = sorted(coherence_values)
            sorted_density = sorted(density_values)

            def median(values: list[float]) -> float:
                middle = len(values) // 2
                return (
                    values[middle]
                    if len(values) % 2
                    else 0.5 * (values[middle - 1] + values[middle])
                )

            if not math.isclose(
                median_coherence,
                median(sorted_coherence),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                median_density,
                median(sorted_density),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"{session_id}: stored median이 segment 재계산과 다릅니다")
            key = (split, family, group, band_index)
            aggregate = group_band_counts.setdefault(key, [0, 0])
            aggregate[0] += total
            aggregate[1] += joint_count

    qualifying: dict[tuple[str, str, int], set[str]] = {}
    for (split, family, group, band_index), (total, joint) in group_band_counts.items():
        if (
            joint >= int(policy["minimum_joint_segments_per_group"])
            and joint / total
            >= float(policy["minimum_joint_segment_fraction_per_group"])
        ):
            qualifying.setdefault((split, family, band_index), set()).add(group)
    summary_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for split in REQUIRED_SPLITS:
        for family in contract.source_families:
            for band_index, band in enumerate(contract.point_control_subbands_hz):
                groups = len(qualifying.get((split, family, band_index), set()))
                passed = groups >= MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND
                summary_rows.append(
                    {
                        "split": split,
                        "source_family": family,
                        "band_hz": [float(value) for value in band],
                        "qualifying_independent_groups": groups,
                        "passed": passed,
                    }
                )
                if not passed:
                    blockers.append(
                        f"{split}/{family}/{band[0]:.0f}-{band[1]:.0f}Hz groups "
                        f"{groups} < {MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND}"
                    )
    expected_summary = {
        "status": "PASS" if not blockers else "BLOCKED",
        "session_count": len(sessions),
        "rows": summary_rows,
        "blockers": blockers,
    }
    if receipt["summary"] != expected_summary:
        raise ValueError("coverage summary가 session evidence 재집계와 다릅니다")
    if blockers:
        raise ValueError("광대역 recorded coverage가 BLOCKED입니다: " + "; ".join(blockers))
    return {
        "status": (
            "PASS" if require_local_files else "STRUCTURAL_ONLY_NOT_CAMPAIGN_ELIGIBLE"
        ),
        "campaign_readiness_eligible": bool(require_local_files),
        "control_band_contract_sha256": contract.digest(),
        "policy_sha256": policy["policy_sha256"],
        "manifest_sha256": manifest["sha256"],
        "plant_evidence_sha256": plant["sha256"],
        "training_timing_contract_sha256": timing_sha,
        "alignment_policy_sha256": alignment_policy_sha,
        "coverage_algorithm_sha256": coverage_algorithm_sha,
        "session_count": len(sessions),
        "rows": summary_rows,
    }


__all__ = [
    "BROADBAND_COVERAGE_RECEIPT_SCHEMA",
    "BROADBAND_SOURCE_TRANSFORM_SCHEMA",
    "MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND",
    "MIN_JOINT_SEGMENTS_PER_GROUP",
    "MIN_JOINT_SEGMENT_FRACTION_PER_GROUP",
    "MIN_SOURCE_ERR_COHERENCE",
    "MIN_TARGET_D_DENSITY_RATIO",
    "NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE",
    "NATIVE_EXACT_TARGET_RATE_ROLE",
    "build_broadband_coverage_policy",
    "minimum_native_sample_rate_hz",
    "seal_broadband_coverage_receipt",
    "sha256_file",
    "validate_broadband_coverage_receipt",
]
