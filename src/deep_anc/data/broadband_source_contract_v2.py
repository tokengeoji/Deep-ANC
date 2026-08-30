"""광대역 recorded-v2 source acquisition 계약 v2.

v1의 무손실·단일 연속 15초 정책을 조용히 완화하지 않는다. v2는 별도 schema에서 다음
두 acquisition 경로를 허용하되 최종 성능/계보 게이트는 그대로 유지한다.

* immutable compressed long-form: 원본 bytes, decoder runtime, full decoded PCM,
  processed PCM/WAV와 최종 Q15를 모두 SHA로 결속한다.
* short-component sequence: 같은 family/split의 서로 다른 lineage component를 최소 3개
  사용하고 모든 component를 하나의 union group으로 취급한다. repeat/loop와 component의
  후보·split 간 재사용은 금지한다.

고정 EQ는 결과별 적응이 아닌 predeclared global/family FIR만 허용한다. EQ 전 각 component가
native 11.314 kHz bandwidth와 일곱 대역 density를 통과해야 하므로 boundary fade나 EQ가 없는
고역을 새로 만든 증거로 사용될 수 없다. 최종 submitted Q15에 대해서도 source/predicted-ERR
9x7 gate를 다시 통과해야 한다.

이 모듈은 오디오 장치를 열지 않는다. ``require_local_files=False`` 검증은 fixture를 포함해
항상 structural-only이며 실제 acquisition PASS나 live authority가 아니다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..dsp.control_band_contract import ControlBandContract
from ..dsp.causal_training_operator import load_causal_training_authority
from ..dsp.fullband_causal_v4 import OPERATOR_NPZ_SCHEMA
from .broadband_recording_campaign import (
    MIN_NATIVE_SAMPLE_RATE_HZ,
    REQUIRED_FAMILIES,
    REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
    REQUIRED_SPLITS,
)
from .broadband_source_inventory import required_campaign_slots
from .recorded_v2_capture import (
    MAX_SUBMITTED_PEAK_INT16,
    SOURCE_FRAMES,
    render_submitted_pcm,
    validate_fullband_causal_plant,
)


BROADBAND_SOURCE_CONTRACT_V2_SCHEMA = "broadband_recorded_v2_source_contract_v2"
BROADBAND_SOURCE_MANIFEST_V2_SCHEMA = (
    "broadband_recorded_v2_source_acquisition_manifest_v2"
)
BROADBAND_SOURCE_MANIFEST_V2_AUDIT_SCHEMA = (
    "broadband_recorded_v2_source_acquisition_audit_v2"
)
BROADBAND_SOURCE_MANIFEST_V2_ISSUED_SCHEMA = (
    "broadband_recorded_v2_source_acquisition_issued_v2"
)

SOURCE_MODES = ("single_long_form", "multi_component_sequence")
EQ_POLICY_MODES = ("none", "global_fixed", "family_fixed")
MIN_SHORT_COMPONENTS = 3
MIN_COMPONENT_PROCESSED_FRAMES = 72_000  # 1.5 s @ 48 kHz
BOUNDARY_FADE_FRAMES = 480  # 10 ms, no-overlap fade-out/fade-in
MAX_COMPONENT_CREST_DB = 15.0
MAX_FINAL_CREST_DB = 15.0
MAX_EQ_BOOST_DB = 12.0
MAX_EQ_ATTENUATION_DB = 12.0
MAX_EQ_CREST_INCREASE_DB = 3.0
MAX_EQ_TAPS = 513
MIN_ALL_BAND_SEGMENTS = 8
MIN_DENSITY_RATIO = 0.25
# production receipt parser/operator schema와 root review가 끝나기 전에는 issuer를 열지 않는다.
SOURCE_MANIFEST_V2_ISSUER_AUTHORITY: dict[str, str] | None = None
SEGMENT_FRAMES = 72_000
SEGMENT_START_FRAMES = tuple(12_000 + SEGMENT_FRAMES * index for index in range(9))
_HEX = frozenset("0123456789abcdef")


class SourceContractV2Blocked(RuntimeError):
    """v2 acquisition evidence가 canonical 발행 조건을 충족하지 못함."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return text


def _commit(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 40 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 full lowercase git SHA가 아닙니다")
    return text


def _exact(value: object, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}가 양의 정수가 아닙니다")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}가 finite number가 아닙니다")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}가 finite number가 아닙니다") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}가 finite number가 아닙니다")
    return number


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _digest(payload)
    return payload


def boundary_fade_coefficients_sha256() -> str:
    """플랫폼 독립 linear Q15 fade-in/out coefficient bytes의 SHA."""

    denominator = BOUNDARY_FADE_FRAMES - 1
    fade_in = [
        (32_767 * index) // denominator for index in range(BOUNDARY_FADE_FRAMES)
    ]
    fade_out = list(reversed(fade_in))
    array = np.asarray([fade_out, fade_in], dtype="<u2")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _boundary_fade_coefficients_float32() -> tuple[np.ndarray, np.ndarray]:
    """계약의 Q15 계수를 실제 composition에 쓰는 exact float32 배열로 만든다."""

    denominator = BOUNDARY_FADE_FRAMES - 1
    fade_in_q15 = np.asarray(
        [(32_767 * index) // denominator for index in range(BOUNDARY_FADE_FRAMES)],
        dtype="<u2",
    )
    fade_out_q15 = fade_in_q15[::-1].copy()
    scale = np.float32(32_767.0)
    return (
        fade_out_q15.astype("<f4") / scale,
        fade_in_q15.astype("<f4") / scale,
    )


def source_contract_v2() -> dict[str, Any]:
    control = ControlBandContract.broadband_point_control()
    value: dict[str, Any] = {
        "schema": BROADBAND_SOURCE_CONTRACT_V2_SCHEMA,
        "v1_unchanged": True,
        "live_authority": None,
        "issuer_authority": SOURCE_MANIFEST_V2_ISSUER_AUTHORITY,
        "source_modes": list(SOURCE_MODES),
        "preference_order": [
            "lossless_single_long_form",
            "immutable_compressed_single_long_form",
            "multi_component_sequence",
        ],
        "minimum_short_components": MIN_SHORT_COMPONENTS,
        "minimum_component_processed_frames": MIN_COMPONENT_PROCESSED_FRAMES,
        "component_reuse_across_candidate_or_split_forbidden": True,
        "minimum_native_sample_rate_hz": MIN_NATIVE_SAMPLE_RATE_HZ,
        "required_native_bandwidth_upper_hz": REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
        "component_pre_eq_all_seven_band_density_minimum": MIN_DENSITY_RATIO,
        "final_per_band_passing_segments_minimum": MIN_ALL_BAND_SEGMENTS,
        "final_density_minimum": MIN_DENSITY_RATIO,
        "final_frames": SOURCE_FRAMES,
        "final_sample_rate_hz": 48_000,
        "final_dtype": "little_endian_int16_mono_raw",
        "final_segment_frames": SEGMENT_FRAMES,
        "final_segment_start_frames": list(SEGMENT_START_FRAMES),
        "joint_causal_operator_npz_schema": OPERATOR_NPZ_SCHEMA,
        "maximum_submitted_peak_int16": MAX_SUBMITTED_PEAK_INT16,
        "maximum_component_crest_db": MAX_COMPONENT_CREST_DB,
        "maximum_final_crest_db": MAX_FINAL_CREST_DB,
        "repeat_or_loop_forbidden": True,
        "composition": {
            "algorithm": "sequential_no_overlap_linear_q15_boundary_fade_v1",
            "boundary_fade_frames": BOUNDARY_FADE_FRAMES,
            "boundary_fade_coefficients_sha256": (
                boundary_fade_coefficients_sha256()
            ),
        },
        "compressed_provenance_required": [
            "immutable_original_file_sha256",
            "codec_and_container",
            "decoder_runtime_fingerprint_sha256",
            "full_decoded_pcm_sha256",
            "processed_pcm_and_wav_sha256",
        ],
        "eq": {
            "modes": list(EQ_POLICY_MODES),
            "only_global_or_family_fixed": True,
            "candidate_adaptive_forbidden": True,
            "maximum_peak_boost_db": MAX_EQ_BOOST_DB,
            "maximum_attenuation_db": MAX_EQ_ATTENUATION_DB,
            "maximum_crest_increase_db": MAX_EQ_CREST_INCREASE_DB,
            "maximum_taps": MAX_EQ_TAPS,
            "policy_must_be_committed_before_candidate_analysis": True,
        },
        "natural_unmodified_level5_challenge_required": True,
        "control_band_contract_sha256": control.digest(),
    }
    value["contract_sha256"] = _digest(value)
    return value


def _file_reference(
    value: object,
    *,
    root: Path | None,
    require_local_files: bool,
    label: str,
) -> dict[str, Any]:
    row = _exact(value, {"path", "size_bytes", "sha256"}, label=label)
    text = str(row["path"])
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}.path가 정규화된 상대경로가 아닙니다")
    size = _positive_int(row["size_bytes"], label=f"{label}.size_bytes")
    digest = _sha(row["sha256"], label=f"{label}.sha256")
    if require_local_files:
        if root is None:
            raise ValueError("local file 검증에 repository root가 없습니다")
        candidate = Path(os.path.abspath(root / path))
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label}.path가 repository 밖입니다") from exc
        current = root
        for part in path.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError(f"{label}.path에 symlink가 있습니다")
        if not candidate.is_file() or candidate.stat().st_size != size:
            raise ValueError(f"{label} local file/size가 다릅니다")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{label} local SHA가 다릅니다")
    return {"path": text, "size_bytes": size, "sha256": digest}


def _density_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{label}는 7대역이어야 합니다")
    result = [_finite(item, label=f"{label} value") for item in value]
    if any(item < MIN_DENSITY_RATIO for item in result):
        raise ValueError(f"{label}가 7대역 density 하한 {MIN_DENSITY_RATIO} 미만입니다")
    return result


def _density_matrix(
    value: object, *, label: str
) -> tuple[list[list[float]], list[int]]:
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError(f"{label}는 9개 segment여야 합니다")
    result = []
    passing = [0] * 7
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 7:
            raise ValueError(f"{label} row가 7대역이 아닙니다")
        row = [_finite(item, label=f"{label} value") for item in raw]
        if any(item < 0.0 for item in row):
            raise ValueError(f"{label}에 음수 density가 있습니다")
        for index, item in enumerate(row):
            if item >= MIN_DENSITY_RATIO:
                passing[index] += 1
        result.append(row)
    if any(count < MIN_ALL_BAND_SEGMENTS for count in passing):
        raise ValueError(
            f"{label} band별 PASS segment가 {MIN_ALL_BAND_SEGMENTS}개 미만입니다: "
            f"counts={passing}"
        )
    return result, passing


def _actual_density_matrix(signal: np.ndarray) -> list[list[float]]:
    """최종 actual Q15 또는 P 적용 결과에서 공식 9x7 density를 재계산."""

    from .broadband_batch_sampler import target_d_density_ratios

    values = np.asarray(signal, dtype=np.float64)
    if values.shape != (SOURCE_FRAMES,) or not np.all(np.isfinite(values)):
        raise ValueError("actual density 입력이 exact 15초 finite mono가 아닙니다")
    bands = ControlBandContract.broadband_point_control().point_control_subbands_hz
    return [
        list(
            target_d_density_ratios(
                values[start : start + SEGMENT_FRAMES],
                sample_rate=48_000,
                bands_hz=bands,
            )
        )
        for start in SEGMENT_START_FRAMES
    ]


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위가 object가 아닙니다: {path}")
    return value


def _apply_physical_primary_operator(
    submitted_q15: np.ndarray,
    *,
    root: Path,
    plant_reference: Mapping[str, Any],
) -> np.ndarray:
    """v4 joint P/S NPZ의 primary view를 actual Q15 전체 history에 적용."""

    from scipy.signal import fftconvolve

    plant_path = root / str(plant_reference["path"])
    plant = _json_no_duplicates(plant_path)
    primary = _exact(
        plant.get("primary_path"),
        {"path", "size_bytes", "sha256"},
        label="physical plant primary_path",
    )
    primary_ref = _file_reference(
        primary,
        root=root,
        require_local_files=True,
        label="physical fullband primary operator",
    )
    secondary_ref = _file_reference(
        _exact(
            plant.get("secondary_path"),
            {"path", "size_bytes", "sha256"},
            label="physical plant secondary_path",
        ),
        root=root,
        require_local_files=True,
        label="physical fullband joint secondary operator",
    )
    if secondary_ref != primary_ref:
        raise SourceContractV2Blocked(
            "physical P/S는 별도 derived NPZ가 아니라 같은 v4 joint NPZ여야 합니다"
        )
    analysis_ref = _file_reference(
        _exact(
            plant.get("analysis"),
            {"path", "size_bytes", "sha256"},
            label="physical plant analysis",
        ),
        root=root,
        require_local_files=True,
        label="physical fullband v4 authority",
    )
    authority_path = root / analysis_ref["path"]
    authority_payload = _json_no_duplicates(authority_path)
    evidence_sha = authority_payload.get("evidence_sha256")
    if not isinstance(evidence_sha, str):
        raise SourceContractV2Blocked("v4 authority evidence SHA가 없습니다")
    try:
        admitted = load_causal_training_authority(
            authority_path,
            expected_file_sha256=analysis_ref["sha256"],
            expected_evidence_sha256=evidence_sha,
            require_live_authority=True,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SourceContractV2Blocked(
            f"physical plant v4 authority 전체 검증 실패: {exc}"
        ) from exc
    operator_path = root / primary_ref["path"]
    if (
        admitted.operator.path != operator_path
        or admitted.operator.file_sha256 != primary_ref["sha256"]
    ):
        raise SourceContractV2Blocked("v4 authority와 physical wrapper joint NPZ가 다릅니다")
    operator = admitted.operator
    fir = operator.primary_post_onset_fir
    delay = operator.primary_coarse_delay_samples
    source = np.asarray(submitted_q15, dtype=np.float64) / 32_767.0
    response = fftconvolve(source, fir, mode="full")[: SOURCE_FRAMES]
    if delay:
        shifted = np.zeros(SOURCE_FRAMES, dtype=np.float64)
        if delay < SOURCE_FRAMES:
            shifted[delay:] = response[: SOURCE_FRAMES - delay]
        response = shifted
    if not np.all(np.isfinite(response)):
        raise SourceContractV2Blocked("physical P 적용 결과가 finite가 아닙니다")
    return np.asarray(response, dtype=np.float64)


def _lineage_union_sha(component_lineage_ids: Sequence[str]) -> str:
    return _digest(sorted(component_lineage_ids))


def _validate_component(
    raw: object,
    *,
    family: str,
    split: str,
    root: Path | None,
    require_local_files: bool,
    label: str,
) -> dict[str, Any]:
    component = _exact(
        raw,
        {
            "component_id",
            "lineage_component_id",
            "source_family",
            "assigned_split",
            "original",
            "decode",
            "processed",
            "excerpt",
            "pre_eq_spectral_crest",
        },
        label=label,
    )
    component_id = str(component["component_id"]).strip()
    lineage_id = str(component["lineage_component_id"]).strip()
    if not component_id or not lineage_id:
        raise ValueError(f"{label} component/lineage id가 비었습니다")
    if component["source_family"] != family or component["assigned_split"] != split:
        raise ValueError(f"{label} component family/split이 campaign slot과 다릅니다")

    original = _exact(
        component["original"],
        {"file", "encoding", "header_receipt"},
        label=f"{label}.original",
    )
    original_file = _file_reference(
        original["file"], root=root, require_local_files=require_local_files,
        label=f"{label}.original.file",
    )
    _file_reference(
        original["header_receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.original.header_receipt",
    )
    encoding = _exact(
        original["encoding"],
        {
            "container",
            "codec",
            "subtype",
            "lossless",
            "native_sample_rate_hz",
            "native_nyquist_hz",
            "channels",
            "frame_count",
        },
        label=f"{label}.original.encoding",
    )
    if not all(str(encoding[key]).strip() for key in ("container", "codec", "subtype")):
        raise ValueError(f"{label} codec/container/subtype가 비었습니다")
    if not isinstance(encoding["lossless"], bool):
        raise ValueError(f"{label} lossless가 bool이 아닙니다")
    native_rate = _positive_int(
        encoding["native_sample_rate_hz"], label=f"{label} native rate"
    )
    native_nyquist = _finite(
        encoding["native_nyquist_hz"], label=f"{label} native Nyquist"
    )
    if (
        native_rate < MIN_NATIVE_SAMPLE_RATE_HZ
        or native_nyquist < REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
        or not math.isclose(native_nyquist, native_rate / 2.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{label} native fs/Nyquist가 8k octave를 덮지 못합니다")
    original_channels = _positive_int(
        encoding["channels"], label=f"{label} original channels"
    )
    if original_channels > 8:
        raise ValueError(f"{label} original channel 수가 1--8 밖입니다")
    native_frames = _positive_int(
        encoding["frame_count"], label=f"{label} native frames"
    )

    decode = _exact(
        component["decode"],
        {
            "decoder_runtime_fingerprint_sha256",
            "decoder_receipt",
            "original_file_sha256",
            "decoded_pcm_file",
            "decoded_pcm_sha256",
            "pcm_dtype",
            "sample_rate_hz",
            "channels",
            "frames",
        },
        label=f"{label}.decode",
    )
    decoder_fingerprint = _sha(
        decode["decoder_runtime_fingerprint_sha256"],
        label=f"{label} decoder fingerprint",
    )
    _file_reference(
        decode["decoder_receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.decode.receipt",
    )
    if _sha(decode["original_file_sha256"], label=f"{label} decode original SHA") != original_file["sha256"]:
        raise ValueError(f"{label} compressed/lossless original→decode SHA가 끊겼습니다")
    decoded_file = _file_reference(
        decode["decoded_pcm_file"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.decode.decoded_pcm_file",
    )
    decoded_sha = _sha(decode["decoded_pcm_sha256"], label=f"{label} decoded PCM SHA")
    if decoded_file["sha256"] != decoded_sha:
        raise ValueError(f"{label} full decoded PCM file/SHA가 다릅니다")
    if (
        decode["pcm_dtype"] != "little_endian_float32_mono_raw"
        or decoded_file["size_bytes"] != native_frames * 4
        or decode["sample_rate_hz"] != native_rate
        or decode["channels"] != 1
        or decode["frames"] != native_frames
    ):
        raise ValueError(f"{label} decoded PCM header lineage가 native header와 다릅니다")

    processed = _exact(
        component["processed"],
        {
            "wav_file",
            "pcm_file",
            "transform_receipt",
            "input_decoded_pcm_sha256",
            "processed_pcm_sha256",
            "pcm_dtype",
            "sample_rate_hz",
            "channels",
            "frames",
            "resample_count",
            "resampler",
        },
        label=f"{label}.processed",
    )
    processed_wav = _file_reference(
        processed["wav_file"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.processed.wav_file",
    )
    processed_pcm = _file_reference(
        processed["pcm_file"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.processed.pcm_file",
    )
    transform_receipt = _file_reference(
        processed["transform_receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.processed.transform_receipt",
    )
    if _sha(processed["input_decoded_pcm_sha256"], label=f"{label} transform input") != decoded_sha:
        raise ValueError(f"{label} decoded→processed SHA가 끊겼습니다")
    processed_pcm_sha = _sha(
        processed["processed_pcm_sha256"], label=f"{label} processed PCM SHA"
    )
    if processed_pcm["sha256"] != processed_pcm_sha:
        raise ValueError(f"{label} processed PCM file/SHA가 다릅니다")
    if (
        processed["pcm_dtype"] != "little_endian_float32_mono_raw"
        or processed["sample_rate_hz"] != 48_000
        or processed["channels"] != 1
    ):
        raise ValueError(f"{label} processed source가 48k mono가 아닙니다")
    processed_frames = _positive_int(
        processed["frames"], label=f"{label} processed frames"
    )
    count = processed["resample_count"]
    if native_rate == 48_000:
        if count != 0 or processed["resampler"] is not None:
            raise ValueError(f"{label} native 48k identity transform가 아닙니다")
    else:
        if count != 1:
            raise ValueError(f"{label} resample은 정확히 1회여야 합니다")
        resampler = _exact(
            processed["resampler"],
            {
                "algorithm",
                "implementation_fingerprint_sha256",
                "frequency_response_receipt",
                "verified_passband_upper_hz",
            },
            label=f"{label}.processed.resampler",
        )
        if resampler["algorithm"] != "polyphase_fir":
            raise ValueError(f"{label} resampler가 polyphase FIR이 아닙니다")
        _sha(
            resampler["implementation_fingerprint_sha256"],
            label=f"{label} resampler fingerprint",
        )
        _file_reference(
            resampler["frequency_response_receipt"], root=root,
            require_local_files=require_local_files,
            label=f"{label}.resampler.frequency_response_receipt",
        )
        if _finite(
            resampler["verified_passband_upper_hz"],
            label=f"{label} resampler passband",
        ) < REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ:
            raise ValueError(f"{label} resampler passband가 11.314kHz 미만입니다")

    excerpt = _exact(
        component["excerpt"],
        {
            "native_start_frame",
            "native_frames",
            "native_excerpt_pcm_file",
            "native_excerpt_pcm_sha256",
            "processed_start_frame",
            "processed_frames",
            "processed_excerpt_pcm_file",
            "processed_excerpt_pcm_sha256",
        },
        label=f"{label}.excerpt",
    )
    native_start = excerpt["native_start_frame"]
    native_excerpt_frames = excerpt["native_frames"]
    processed_start = excerpt["processed_start_frame"]
    excerpt_frames = excerpt["processed_frames"]
    for value, name in (
        (native_start, "native start"),
        (processed_start, "processed start"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} {name}가 0 이상 정수가 아닙니다")
    native_excerpt_frames = _positive_int(
        native_excerpt_frames, label=f"{label} native excerpt frames"
    )
    excerpt_frames = _positive_int(
        excerpt_frames, label=f"{label} processed excerpt frames"
    )
    if (
        native_start + native_excerpt_frames > native_frames
        or processed_start + excerpt_frames > processed_frames
        or excerpt_frames < MIN_COMPONENT_PROCESSED_FRAMES
    ):
        raise ValueError(f"{label} excerpt 범위/최소 1.5초가 유효하지 않습니다")
    native_excerpt_sha = _sha(
        excerpt["native_excerpt_pcm_sha256"],
        label=f"{label} native excerpt PCM SHA",
    )
    native_excerpt_file = _file_reference(
        excerpt["native_excerpt_pcm_file"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.excerpt.native_pcm_file",
    )
    if (
        native_excerpt_file["sha256"] != native_excerpt_sha
        or native_excerpt_file["size_bytes"] != native_excerpt_frames * 4
    ):
        raise ValueError(f"{label} native excerpt file/SHA/size가 다릅니다")
    excerpt_file = _file_reference(
        excerpt["processed_excerpt_pcm_file"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.excerpt.processed_pcm_file",
    )
    excerpt_sha = _sha(
        excerpt["processed_excerpt_pcm_sha256"],
        label=f"{label} processed excerpt PCM SHA",
    )
    if excerpt_file["sha256"] != excerpt_sha:
        raise ValueError(f"{label} processed excerpt file/SHA가 다릅니다")
    if excerpt_file["size_bytes"] != excerpt_frames * 4:
        raise ValueError(f"{label} processed excerpt float32 size가 다릅니다")

    spectrum = _exact(
        component["pre_eq_spectral_crest"],
        {
            "receipt",
            "decoded_pcm_sha256",
            "native_excerpt_pcm_sha256",
            "processed_excerpt_pcm_sha256",
            "control_band_contract_sha256",
            "point_control_subbands_hz",
            "analysed_upper_hz",
            "actual_native_bandwidth_verified",
            "density_ratios_7",
            "crest_factor_db",
            "boundary_or_eq_used_for_evidence",
        },
        label=f"{label}.pre_eq_spectral_crest",
    )
    spectral_receipt = _file_reference(
        spectrum["receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{label}.pre_eq_spectral_crest.receipt",
    )
    if (
        _sha(spectrum["decoded_pcm_sha256"], label=f"{label} spectrum decoded SHA")
        != decoded_sha
        or _sha(
            spectrum["native_excerpt_pcm_sha256"],
            label=f"{label} spectrum native excerpt SHA",
        )
        != native_excerpt_sha
        or _sha(
            spectrum["processed_excerpt_pcm_sha256"],
            label=f"{label} spectrum processed excerpt SHA",
        )
        != excerpt_sha
    ):
        raise ValueError(f"{label} component spectral evidence의 PCM lineage가 다릅니다")
    control = ControlBandContract.broadband_point_control()
    expected_bands = [list(band) for band in control.point_control_subbands_hz]
    if (
        spectrum["control_band_contract_sha256"] != control.digest()
        or spectrum["point_control_subbands_hz"] != expected_bands
    ):
        raise ValueError(f"{label} component spectral control-band가 다릅니다")
    if (
        _finite(spectrum["analysed_upper_hz"], label=f"{label} spectrum upper")
        < REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
        or spectrum["actual_native_bandwidth_verified"] is not True
        or spectrum["boundary_or_eq_used_for_evidence"] is not False
    ):
        raise ValueError(f"{label} native pre-EQ 11.314kHz evidence가 없습니다")
    stored_component_density = _density_vector(
        spectrum["density_ratios_7"], label=f"{label} component density"
    )
    crest = _finite(spectrum["crest_factor_db"], label=f"{label} component crest")
    if crest < 0.0 or crest > MAX_COMPONENT_CREST_DB:
        raise ValueError(f"{label} component crest가 0--15dB 밖입니다")
    if require_local_files and root is not None:
        import soundfile as sf

        decoded_values = np.fromfile(root / decoded_file["path"], dtype="<f4")
        processed_values = np.fromfile(root / processed_pcm["path"], dtype="<f4")
        native_excerpt_values = np.fromfile(
            root / native_excerpt_file["path"], dtype="<f4"
        )
        processed_excerpt_values = np.fromfile(
            root / excerpt_file["path"], dtype="<f4"
        )
        if (
            decoded_values.shape != (native_frames,)
            or processed_values.shape != (processed_frames,)
            or not np.array_equal(
                native_excerpt_values,
                decoded_values[native_start : native_start + native_excerpt_frames],
            )
            or not np.array_equal(
                processed_excerpt_values,
                processed_values[processed_start : processed_start + excerpt_frames],
            )
        ):
            raise ValueError(f"{label} decoded/processed/excerpt 실제 PCM slice 관계가 다릅니다")
        wav_info = sf.info(str(root / processed_wav["path"]))
        wav_values, wav_rate = sf.read(
            str(root / processed_wav["path"]), dtype="float32", always_2d=False
        )
        if (
            wav_rate != 48_000
            or wav_info.channels != 1
            or wav_info.frames != processed_frames
            or np.asarray(wav_values).shape != (processed_frames,)
            or not np.array_equal(np.asarray(wav_values, dtype="<f4"), processed_values)
        ):
            raise ValueError(f"{label} processed WAV↔PCM 실제 bytes 관계가 다릅니다")
        from .broadband_batch_sampler import target_d_density_ratios

        actual_component_density = list(
            target_d_density_ratios(
                native_excerpt_values,
                sample_rate=native_rate,
                bands_hz=control.point_control_subbands_hz,
            )
        )
        native_values_f64 = native_excerpt_values.astype(np.float64)
        actual_peak = float(np.max(np.abs(native_values_f64)))
        actual_rms = float(np.sqrt(np.mean(native_values_f64 * native_values_f64)))
        actual_component_crest = (
            20.0 * math.log10(actual_peak / actual_rms)
            if actual_rms > 0.0
            else math.inf
        )
        if actual_component_density != stored_component_density or not math.isclose(
            actual_component_crest, crest, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"{label} actual native excerpt spectral/crest 재계산이 JSON과 다릅니다"
            )
    return {
        "component_id": component_id,
        "lineage_component_id": lineage_id,
        "lossless": bool(encoding["lossless"]),
        "original_file_sha256": original_file["sha256"],
        "decoder_runtime_fingerprint_sha256": decoder_fingerprint,
        "decoded_pcm_sha256": decoded_sha,
        "processed_wav_sha256": processed_wav["sha256"],
        "processed_pcm_sha256": processed_pcm_sha,
        "processed_excerpt_pcm_sha256": excerpt_sha,
        "processed_excerpt_path": excerpt_file["path"],
        "processed_excerpt_frames": excerpt_frames,
        "transform_receipt_sha256": transform_receipt["sha256"],
        "spectral_receipt_sha256": spectral_receipt["sha256"],
    }


def _validate_eq_policy_set(
    raw: object,
    *,
    root: Path | None,
    require_local_files: bool,
) -> tuple[str, dict[str, dict[str, Any]]]:
    value = _exact(
        raw,
        {
            "mode",
            "predeclared_policy_commit_sha",
            "candidate_analysis_commit_sha",
            "ancestry_receipt",
            "authority_file",
            "policies",
        },
        label="v2 eq policy set",
    )
    mode = str(value["mode"])
    if mode not in EQ_POLICY_MODES:
        raise ValueError("v2 EQ policy mode가 canonical 집합 밖입니다")
    policy_commit = _commit(
        value["predeclared_policy_commit_sha"], label="EQ policy commit"
    )
    analysis_commit = _commit(
        value["candidate_analysis_commit_sha"], label="candidate analysis commit"
    )
    if policy_commit == analysis_commit:
        raise ValueError("EQ policy는 candidate analysis보다 앞선 별도 commit이어야 합니다")
    _file_reference(
        value["ancestry_receipt"], root=root,
        require_local_files=require_local_files,
        label="EQ policy ancestry receipt",
    )
    _file_reference(
        value["authority_file"], root=root,
        require_local_files=require_local_files,
        label="EQ policy authority file",
    )
    policies = value["policies"]
    if not isinstance(policies, list):
        raise ValueError("EQ policies가 list가 아닙니다")
    expected_count = 1 if mode in {"none", "global_fixed"} else len(REQUIRED_FAMILIES)
    if len(policies) != expected_count:
        raise ValueError("EQ policy 수가 global/family scope와 다릅니다")
    result: dict[str, dict[str, Any]] = {}
    seen_families: set[str] = set()
    for index, raw_policy in enumerate(policies):
        policy = _exact(
            raw_policy,
            {
                "policy_id",
                "scope",
                "source_family",
                "fir_coefficients_sha256",
                "frequency_response_receipt",
                "taps",
                "actual_peak_boost_db",
                "actual_max_attenuation_db",
                "maximum_crest_increase_db",
                "adaptive_to_candidate",
            },
            label=f"EQ policy#{index}",
        )
        policy_id = str(policy["policy_id"]).strip()
        if not policy_id or policy_id in result:
            raise ValueError("EQ policy id가 비었거나 중복입니다")
        scope = str(policy["scope"])
        family = policy["source_family"]
        if mode in {"none", "global_fixed"}:
            if scope != "global" or family is not None:
                raise ValueError("global EQ policy scope/family가 다릅니다")
        else:
            family = str(family)
            if scope != "family" or family not in REQUIRED_FAMILIES or family in seen_families:
                raise ValueError("family EQ policy가 family별 정확히 하나가 아닙니다")
            seen_families.add(family)
        _sha(policy["fir_coefficients_sha256"], label=f"{policy_id} FIR SHA")
        _file_reference(
            policy["frequency_response_receipt"], root=root,
            require_local_files=require_local_files,
            label=f"{policy_id} response receipt",
        )
        taps = _positive_int(policy["taps"], label=f"{policy_id} taps")
        boost = _finite(policy["actual_peak_boost_db"], label=f"{policy_id} boost")
        attenuation = _finite(
            policy["actual_max_attenuation_db"], label=f"{policy_id} attenuation"
        )
        crest_limit = _finite(
            policy["maximum_crest_increase_db"], label=f"{policy_id} crest limit"
        )
        if (
            taps > MAX_EQ_TAPS
            or boost < 0.0
            or boost > MAX_EQ_BOOST_DB
            or attenuation < 0.0
            or attenuation > MAX_EQ_ATTENUATION_DB
            or crest_limit < 0.0
            or crest_limit > MAX_EQ_CREST_INCREASE_DB
            or policy["adaptive_to_candidate"] is not False
        ):
            raise ValueError(f"{policy_id} EQ가 fixed bounded 계약 밖입니다")
        if mode == "none" and (taps != 1 or boost != 0.0 or attenuation != 0.0):
            raise ValueError("EQ none policy가 identity가 아닙니다")
        result[policy_id] = {
            "mode": mode,
            "family": family,
            "boost": boost,
            "attenuation": attenuation,
            "crest_limit": crest_limit,
        }
    if mode == "family_fixed" and seen_families != set(REQUIRED_FAMILIES):
        raise ValueError("family fixed EQ가 네 family를 모두 갖지 않습니다")
    if require_local_files and root is not None:
        for commit in (policy_commit, analysis_commit):
            checked = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            if checked.returncode != 0:
                raise ValueError("EQ policy/analysis commit이 local git object가 아닙니다")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", policy_commit, analysis_commit],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("EQ policy commit이 analysis commit의 ancestor가 아닙니다")
    return mode, result


def _validate_candidate(
    raw: object,
    *,
    slot: Mapping[str, str],
    plant_reference: Mapping[str, Any],
    eq_mode: str,
    policies: Mapping[str, Mapping[str, Any]],
    root: Path | None,
    require_local_files: bool,
) -> dict[str, Any]:
    plant_sha256 = str(plant_reference["sha256"])
    candidate = _exact(
        raw,
        {
            "slot_id",
            "candidate_id",
            "split",
            "source_family",
            "mode",
            "components",
            "composition",
            "lineage_union",
            "selection_evidence",
            "eq_transform",
            "final_submission",
            "corpus_disjointness",
        },
        label=f"v2 candidate {slot['slot_id']}",
    )
    if (
        candidate["slot_id"] != slot["slot_id"]
        or candidate["split"] != slot["split"]
        or candidate["source_family"] != slot["source_family"]
    ):
        raise ValueError(f"{slot['slot_id']} candidate slot/split/family가 다릅니다")
    candidate_id = str(candidate["candidate_id"]).strip()
    mode = str(candidate["mode"])
    if not candidate_id or mode not in SOURCE_MODES:
        raise ValueError(f"{slot['slot_id']} candidate id/mode가 유효하지 않습니다")
    raw_components = candidate["components"]
    if not isinstance(raw_components, list):
        raise ValueError(f"{candidate_id} components가 list가 아닙니다")
    expected_count = 1 if mode == "single_long_form" else MIN_SHORT_COMPONENTS
    if len(raw_components) < expected_count or (
        mode == "single_long_form" and len(raw_components) != 1
    ):
        raise ValueError(f"{candidate_id} mode별 component 수가 다릅니다")
    components = [
        _validate_component(
            component,
            family=slot["source_family"],
            split=slot["split"],
            root=root,
            require_local_files=require_local_files,
            label=f"{candidate_id}.component#{index}",
        )
        for index, component in enumerate(raw_components)
    ]
    uniqueness_fields = (
        "component_id",
        "lineage_component_id",
        "original_file_sha256",
        "decoded_pcm_sha256",
        "processed_pcm_sha256",
        "processed_excerpt_pcm_sha256",
    )
    for field in uniqueness_fields:
        values = [component[field] for component in components]
        if len(values) != len(set(values)):
            raise ValueError(f"{candidate_id} 안에서 {field}를 반복했습니다")

    composition = _exact(
        candidate["composition"],
        {
            "algorithm",
            "ordered_component_ids",
            "output_frames",
            "boundaries",
            "pre_eq_pcm_sha256",
            "recipe_receipt",
        },
        label=f"{candidate_id}.composition",
    )
    if composition["algorithm"] != source_contract_v2()["composition"]["algorithm"]:
        raise ValueError(f"{candidate_id} composition algorithm이 다릅니다")
    ordered = [component["component_id"] for component in components]
    if composition["ordered_component_ids"] != ordered:
        raise ValueError(f"{candidate_id} component 순서가 receipt와 다릅니다")
    if composition["output_frames"] != SOURCE_FRAMES or sum(
        component["processed_excerpt_frames"] for component in components
    ) != SOURCE_FRAMES:
        raise ValueError(f"{candidate_id} composition이 exact 15초가 아닙니다")
    pre_eq_sha = _sha(
        composition["pre_eq_pcm_sha256"], label=f"{candidate_id} pre-EQ PCM SHA"
    )
    _file_reference(
        composition["recipe_receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.composition.recipe_receipt",
    )
    boundaries = composition["boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != len(components) - 1:
        raise ValueError(f"{candidate_id} boundary 수가 component 수와 다릅니다")
    cumulative = 0
    expected_coefficients = boundary_fade_coefficients_sha256()
    for index, boundary_raw in enumerate(boundaries):
        cumulative += components[index]["processed_excerpt_frames"]
        boundary = _exact(
            boundary_raw,
            {
                "left_component_id",
                "right_component_id",
                "output_frame",
                "fade_frames_each_side",
                "coefficient_q15_sha256",
                "receipt",
            },
            label=f"{candidate_id}.boundary#{index}",
        )
        if (
            boundary["left_component_id"] != ordered[index]
            or boundary["right_component_id"] != ordered[index + 1]
            or boundary["output_frame"] != cumulative
            or boundary["fade_frames_each_side"] != BOUNDARY_FADE_FRAMES
            or boundary["coefficient_q15_sha256"] != expected_coefficients
        ):
            raise ValueError(f"{candidate_id} boundary fade receipt가 다릅니다")
        _file_reference(
            boundary["receipt"], root=root,
            require_local_files=require_local_files,
            label=f"{candidate_id}.boundary#{index}.receipt",
        )
    if mode == "single_long_form" and boundaries:
        raise ValueError(f"{candidate_id} single source에 boundary가 있습니다")

    lineage = _exact(
        candidate["lineage_union"],
        {
            "component_lineage_ids",
            "union_identity_sha256",
            "dsu_authority_sha256",
            "all_components_same_family",
            "no_component_reuse_across_candidates_or_splits",
        },
        label=f"{candidate_id}.lineage_union",
    )
    lineage_ids = sorted(component["lineage_component_id"] for component in components)
    if (
        lineage["component_lineage_ids"] != lineage_ids
        or lineage["union_identity_sha256"] != _lineage_union_sha(lineage_ids)
        or lineage["all_components_same_family"] is not True
        or lineage["no_component_reuse_across_candidates_or_splits"] is not True
    ):
        raise ValueError(f"{candidate_id} lineage union identity가 다릅니다")
    _sha(lineage["dsu_authority_sha256"], label=f"{candidate_id} DSU authority")

    lossless_single = mode == "single_long_form" and components[0]["lossless"]
    rank = 0 if lossless_single else 1 if mode == "single_long_form" else 2
    selection = _exact(
        candidate["selection_evidence"],
        {
            "selected_preference_rank",
            "eligible_better_rank_candidate_count",
            "inventory_authority_sha256",
            "receipt",
            "reason",
        },
        label=f"{candidate_id}.selection_evidence",
    )
    if (
        selection["selected_preference_rank"] != rank
        or selection["eligible_better_rank_candidate_count"] != 0
        or not str(selection["reason"]).strip()
    ):
        raise ValueError(f"{candidate_id} lossless long-form 우선순위 증거가 없습니다")
    _sha(selection["inventory_authority_sha256"], label=f"{candidate_id} inventory SHA")
    _file_reference(
        selection["receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.selection.receipt",
    )

    eq = _exact(
        candidate["eq_transform"],
        {
            "policy_id",
            "input_pre_eq_pcm_sha256",
            "output_post_eq_pcm_sha256",
            "receipt",
            "actual_crest_increase_db",
            "adaptive_to_candidate",
        },
        label=f"{candidate_id}.eq_transform",
    )
    policy_id = str(eq["policy_id"])
    policy = policies.get(policy_id)
    if policy is None:
        raise ValueError(f"{candidate_id} EQ policy id가 authority에 없습니다")
    if eq_mode == "family_fixed" and policy["family"] != slot["source_family"]:
        raise ValueError(f"{candidate_id} family EQ policy가 family와 다릅니다")
    if (
        _sha(eq["input_pre_eq_pcm_sha256"], label=f"{candidate_id} EQ input")
        != pre_eq_sha
        or eq["adaptive_to_candidate"] is not False
    ):
        raise ValueError(f"{candidate_id} EQ input/adaptive 계약이 다릅니다")
    post_eq_sha = _sha(
        eq["output_post_eq_pcm_sha256"], label=f"{candidate_id} post-EQ PCM SHA"
    )
    crest_increase = _finite(
        eq["actual_crest_increase_db"], label=f"{candidate_id} EQ crest increase"
    )
    if crest_increase < 0.0 or crest_increase > min(
        policy["crest_limit"], MAX_EQ_CREST_INCREASE_DB
    ):
        raise ValueError(f"{candidate_id} EQ crest 증가가 bounded policy 밖입니다")
    if eq_mode == "none" and (post_eq_sha != pre_eq_sha or crest_increase != 0.0):
        raise ValueError(f"{candidate_id} identity EQ가 bytes/crest를 바꿨습니다")
    _file_reference(
        eq["receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.eq_transform.receipt",
    )

    final = _exact(
        candidate["final_submission"],
        {
            "role",
            "post_eq_pcm_sha256",
            "processed_wav_file",
            "submitted_q15le_file",
            "submitted_q15_pcm_sha256",
            "sample_rate_hz",
            "frames",
            "dtype",
            "gain_q15",
            "peak_int16",
            "crest_factor_db",
            "control_band_contract_sha256",
            "point_control_subbands_hz",
            "segment_start_frames",
            "segment_frames",
            "source_density_ratios_9x7",
            "source_density_pass_counts_7",
            "predicted_err_density_ratios_9x7",
            "predicted_err_density_pass_counts_7",
            "spectral_crest_receipt",
            "canonical_fullband_plant_evidence_sha256",
            "unmodified_natural_challenge",
        },
        label=f"{candidate_id}.final_submission",
    )
    if final["role"] != "coverage_source_not_unmodified_level5_challenge":
        raise ValueError(f"{candidate_id} shaped coverage와 natural challenge 역할이 섞였습니다")
    if _sha(final["post_eq_pcm_sha256"], label=f"{candidate_id} final post-EQ SHA") != post_eq_sha:
        raise ValueError(f"{candidate_id} EQ→final SHA가 끊겼습니다")
    final_wav = _file_reference(
        final["processed_wav_file"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.final.processed_wav",
    )
    submitted = _file_reference(
        final["submitted_q15le_file"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.final.submitted_q15le",
    )
    submitted_sha = _sha(
        final["submitted_q15_pcm_sha256"], label=f"{candidate_id} submitted Q15 SHA"
    )
    if submitted["sha256"] != submitted_sha or submitted["size_bytes"] != SOURCE_FRAMES * 2:
        raise ValueError(f"{candidate_id} submitted Q15 file/SHA/size가 다릅니다")
    if (
        final["sample_rate_hz"] != 48_000
        or final["frames"] != SOURCE_FRAMES
        or final["dtype"] != "little_endian_int16_mono_raw"
    ):
        raise ValueError(f"{candidate_id} final Q15가 exact 15초 48k가 아닙니다")
    gain_q15 = final["gain_q15"]
    if (
        isinstance(gain_q15, bool)
        or not isinstance(gain_q15, int)
        or not 1 <= gain_q15 <= 32_767
    ):
        raise ValueError(f"{candidate_id} final gain_q15이 유효하지 않습니다")
    control = ControlBandContract.broadband_point_control()
    if (
        final["control_band_contract_sha256"] != control.digest()
        or final["point_control_subbands_hz"]
        != [list(band) for band in control.point_control_subbands_hz]
        or final["segment_start_frames"] != list(SEGMENT_START_FRAMES)
        or final["segment_frames"] != SEGMENT_FRAMES
    ):
        raise ValueError(f"{candidate_id} final 9x7 population/control-band가 다릅니다")
    peak = final["peak_int16"]
    if isinstance(peak, bool) or not isinstance(peak, int) or not 1 <= peak <= MAX_SUBMITTED_PEAK_INT16:
        raise ValueError(f"{candidate_id} final peak가 안전 범위 밖입니다")
    final_crest = _finite(final["crest_factor_db"], label=f"{candidate_id} final crest")
    if final_crest < 0.0 or final_crest > MAX_FINAL_CREST_DB:
        raise ValueError(f"{candidate_id} final crest가 0--15dB 밖입니다")
    stored_source_matrix, source_pass = _density_matrix(
        final["source_density_ratios_9x7"], label=f"{candidate_id} final source density"
    )
    stored_predicted_matrix, predicted_pass = _density_matrix(
        final["predicted_err_density_ratios_9x7"],
        label=f"{candidate_id} final predicted ERR density",
    )
    if final["source_density_pass_counts_7"] != source_pass:
        raise ValueError(f"{candidate_id} final source band별 PASS count가 matrix와 다릅니다")
    if final["predicted_err_density_pass_counts_7"] != predicted_pass:
        raise ValueError(
            f"{candidate_id} final predicted ERR band별 PASS count가 matrix와 다릅니다"
        )
    _file_reference(
        final["spectral_crest_receipt"], root=root,
        require_local_files=require_local_files,
        label=f"{candidate_id}.final.spectral_crest_receipt",
    )
    if (
        _sha(
            final["canonical_fullband_plant_evidence_sha256"],
            label=f"{candidate_id} final plant SHA",
        )
        != plant_sha256
        or final["unmodified_natural_challenge"] is not False
    ):
        raise ValueError(f"{candidate_id} plant/challenge role가 다릅니다")
    if require_local_files and root is not None:
        import soundfile as sf

        samples = np.fromfile(root / submitted["path"], dtype="<i2")
        if samples.shape != (SOURCE_FRAMES,):
            raise ValueError(f"{candidate_id} submitted Q15 실제 shape가 다릅니다")
        values = samples.astype(np.float64)
        actual_peak = int(np.max(np.abs(samples.astype(np.int32))))
        rms = float(np.sqrt(np.mean(values * values)))
        actual_crest = 20.0 * math.log10(actual_peak / rms) if rms > 0.0 else math.inf
        if actual_peak != peak or not math.isclose(
            actual_crest, final_crest, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{candidate_id} submitted Q15 peak/crest 재계산이 다릅니다")
        processed_wav_path = root / final_wav["path"]
        wav_info = sf.info(str(processed_wav_path))
        post_eq_values, post_eq_rate = sf.read(
            str(processed_wav_path), dtype="float32", always_2d=False
        )
        post_eq_values = np.ascontiguousarray(
            np.asarray(post_eq_values, dtype="<f4")
        )
        if (
            post_eq_rate != 48_000
            or wav_info.channels != 1
            or wav_info.frames != SOURCE_FRAMES
            or post_eq_values.shape != (SOURCE_FRAMES,)
            or hashlib.sha256(post_eq_values.tobytes(order="C")).hexdigest()
            != post_eq_sha
        ):
            raise ValueError(f"{candidate_id} final processed WAV↔post-EQ PCM SHA가 다릅니다")
        if eq_mode != "none":
            raise SourceContractV2Blocked(
                f"{candidate_id} fixed EQ actual FIR 적용 검증 authority가 아직 닫혀 있습니다"
            )
        excerpts = [
            np.fromfile(root / component["processed_excerpt_path"], dtype="<f4")
            for component in components
        ]
        reconstructed_pre_eq = np.ascontiguousarray(
            np.concatenate(excerpts), dtype="<f4"
        )
        if reconstructed_pre_eq.shape != (SOURCE_FRAMES,):
            raise ValueError(f"{candidate_id} actual composition 길이가 exact 15초가 아닙니다")
        fade_out, fade_in = _boundary_fade_coefficients_float32()
        boundary_frame = 0
        for index in range(len(excerpts) - 1):
            boundary_frame += int(excerpts[index].size)
            reconstructed_pre_eq[
                boundary_frame - BOUNDARY_FADE_FRAMES : boundary_frame
            ] *= fade_out
            reconstructed_pre_eq[
                boundary_frame : boundary_frame + BOUNDARY_FADE_FRAMES
            ] *= fade_in
        actual_pre_eq_sha = hashlib.sha256(
            reconstructed_pre_eq.tobytes(order="C")
        ).hexdigest()
        if actual_pre_eq_sha != pre_eq_sha or not np.array_equal(
            reconstructed_pre_eq, post_eq_values
        ):
            raise ValueError(
                f"{candidate_id} component excerpts→composition→identity EQ bytes가 다릅니다"
            )
        expected_q15, _ = render_submitted_pcm(
            processed_wav_path,
            start_frame=0,
            gain_q15=gain_q15,
        )
        if not np.array_equal(expected_q15, samples):
            raise ValueError(f"{candidate_id} processed WAV→submitted Q15 quantization이 다릅니다")
        actual_source_matrix = _actual_density_matrix(samples.astype(np.float64))
        if actual_source_matrix != stored_source_matrix:
            raise ValueError(f"{candidate_id} actual Q15 source 9x7 재계산이 JSON과 다릅니다")
        predicted_err = _apply_physical_primary_operator(
            samples,
            root=root,
            plant_reference=plant_reference,
        )
        actual_predicted_matrix = _actual_density_matrix(predicted_err)
        if actual_predicted_matrix != stored_predicted_matrix:
            raise ValueError(f"{candidate_id} actual P-applied predicted ERR 9x7이 JSON과 다릅니다")

    disjoint = _exact(
        candidate["corpus_disjointness"],
        {
            "authority_sha256",
            "component_ids",
            "all_raw_content_disjoint",
            "all_decoded_and_processed_content_disjoint",
            "all_lineage_components_disjoint",
        },
        label=f"{candidate_id}.corpus_disjointness",
    )
    if (
        disjoint["component_ids"] != sorted(ordered)
        or any(
            disjoint[key] is not True
            for key in (
                "all_raw_content_disjoint",
                "all_decoded_and_processed_content_disjoint",
                "all_lineage_components_disjoint",
            )
        )
    ):
        raise ValueError(f"{candidate_id} 기존 corpus disjointness가 없습니다")
    _sha(disjoint["authority_sha256"], label=f"{candidate_id} disjointness SHA")
    return {
        "candidate_id": candidate_id,
        "slot_id": slot["slot_id"],
        "split": slot["split"],
        "source_family": slot["source_family"],
        "mode": mode,
        "selection_rank": rank,
        "eq_policy_id": policy_id,
        "lineage_union_identity_sha256": lineage["union_identity_sha256"],
        "component_count": len(components),
        "source_per_band_passing_segments": source_pass,
        "predicted_err_per_band_passing_segments": predicted_pass,
        "components": components,
        "submitted_q15_pcm_sha256": submitted_sha,
    }


def validate_source_manifest_v2(
    manifest: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    repository_root: str | Path | None = None,
    require_local_files: bool = False,
) -> dict[str, Any]:
    """v2 manifest를 검증한다. metadata-only는 절대 실제 PASS가 아니다."""

    root = None
    if repository_root is not None:
        root = Path(repository_root).resolve(strict=True)
    value = _exact(
        manifest,
        {
            "schema",
            "role",
            "status",
            "synthetic_fixture",
            "contract_sha256",
            "control_band_contract_sha256",
            "campaign_evidence_sha256",
            "physical_fullband_plant_evidence",
            "eq_policy_set",
            "selection_inventory_authority_sha256",
            "unmodified_level5_challenge_required",
            "candidates",
            "evidence_sha256",
        },
        label="source acquisition manifest v2",
    )
    if (
        value["schema"] != BROADBAND_SOURCE_MANIFEST_V2_SCHEMA
        or value["role"] != "candidate_evidence_not_live_source_plan"
        or value["status"] != "DRAFT"
    ):
        raise ValueError("source manifest v2 schema/role/status가 DRAFT 계약과 다릅니다")
    contract = source_contract_v2()
    control = ControlBandContract.broadband_point_control()
    if (
        value["contract_sha256"] != contract["contract_sha256"]
        or value["control_band_contract_sha256"] != control.digest()
        or value["campaign_evidence_sha256"] != campaign.get("evidence_sha256")
    ):
        raise ValueError("source manifest v2 contract/campaign SHA가 다릅니다")
    if value["unmodified_level5_challenge_required"] is not True:
        raise ValueError("unmodified Level-5 challenge를 끌 수 없습니다")
    _sha(
        value["selection_inventory_authority_sha256"],
        label="selection inventory authority SHA",
    )
    stored_evidence = _sha(value["evidence_sha256"], label="manifest v2 evidence SHA")
    expected_evidence = _digest(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    if stored_evidence != expected_evidence:
        raise ValueError("source manifest v2 evidence SHA가 payload와 다릅니다")
    if require_local_files and value["synthetic_fixture"] is not False:
        raise SourceContractV2Blocked("synthetic fixture는 acquisition issuer에 사용할 수 없습니다")
    plant_ref = _file_reference(
        value["physical_fullband_plant_evidence"], root=root,
        require_local_files=require_local_files,
        label="source manifest v2 physical plant",
    )
    if require_local_files:
        if root is None:
            raise ValueError("local publisher 검증에 repository root가 없습니다")
        validate_fullband_causal_plant(
            root / plant_ref["path"],
            expected_file_sha256=plant_ref["sha256"],
            repository_root=root,
        )
    eq_mode, policies = _validate_eq_policy_set(
        value["eq_policy_set"], root=root,
        require_local_files=require_local_files,
    )
    slots = required_campaign_slots(campaign)
    slot_by_id = {row["slot_id"]: row for row in slots}
    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("source manifest v2 candidates가 list가 아닙니다")
    by_slot: dict[str, Any] = {}
    for raw_candidate in raw_candidates:
        slot_id = str(raw_candidate.get("slot_id", "")) if isinstance(raw_candidate, Mapping) else ""
        if slot_id not in slot_by_id or slot_id in by_slot:
            raise ValueError("source manifest v2 candidate slot이 unknown/duplicate입니다")
        by_slot[slot_id] = raw_candidate

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for slot in slots:
        raw_candidate = by_slot.get(slot["slot_id"])
        if raw_candidate is None:
            rejected.append(
                {"slot_id": slot["slot_id"], "reason": "candidate_missing"}
            )
            continue
        try:
            valid.append(
                _validate_candidate(
                    raw_candidate,
                    slot=slot,
                    plant_reference=plant_ref,
                    eq_mode=eq_mode,
                    policies=policies,
                    root=root,
                    require_local_files=require_local_files,
                )
            )
        except (SourceContractV2Blocked, TypeError, ValueError) as exc:
            rejected.append({"slot_id": slot["slot_id"], "reason": str(exc)})

    # 어떤 original/decoded/processed/excerpt/lineage component도 후보나 split을 넘어 재사용할
    # 수 없다. 한 composite의 union identity만 달리 써서 재사용하는 우회도 모두 잡는다.
    global_fields = (
        "component_id",
        "lineage_component_id",
        "original_file_sha256",
        "decoded_pcm_sha256",
        "processed_pcm_sha256",
        "processed_excerpt_pcm_sha256",
    )
    owners: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in valid:
        for component in row["components"]:
            for field in global_fields:
                owners[(field, str(component[field]))].append(row["slot_id"])
    duplicate_slots: defaultdict[str, list[str]] = defaultdict(list)
    for (field, _), slot_ids in owners.items():
        if len(slot_ids) > 1:
            for slot_id in slot_ids:
                duplicate_slots[slot_id].append(f"component_reused:{field}")
    if duplicate_slots:
        kept = []
        for row in valid:
            reasons = duplicate_slots.get(row["slot_id"])
            if reasons:
                rejected.append(
                    {"slot_id": row["slot_id"], "reason": ",".join(sorted(set(reasons)))}
                )
            else:
                kept.append(row)
        valid = kept

    counts = Counter((row["split"], row["source_family"]) for row in valid)
    cells = []
    deficit = 0
    for split in REQUIRED_SPLITS:
        for family in REQUIRED_FAMILIES:
            count = counts[(split, family)]
            missing = max(0, 4 - count)
            deficit += missing
            cells.append(
                {
                    "split": split,
                    "source_family": family,
                    "structurally_valid_candidates": count,
                    "required": 4,
                    "deficit": missing,
                }
            )
    structurally_complete = deficit == 0 and not rejected and len(valid) == 48
    locally_rechecked = structurally_complete and require_local_files
    issuer_enabled = SOURCE_MANIFEST_V2_ISSUER_AUTHORITY is not None
    locally_verified = locally_rechecked and issuer_enabled
    if locally_verified:
        status = "VERIFIED_48_LOCAL_NOT_LIVE_PLAN"
    elif locally_rechecked:
        status = "LOCAL_EVIDENCE_RECHECKED_ISSUER_BLOCKED"
    elif structurally_complete:
        status = "STRUCTURAL_ONLY_NOT_PUBLISHABLE"
    else:
        status = "BLOCKED"
    result = {
        "schema": BROADBAND_SOURCE_MANIFEST_V2_AUDIT_SCHEMA,
        "status": status,
        "actual_acquisition_pass": locally_verified,
        "issuer_authority": SOURCE_MANIFEST_V2_ISSUER_AUTHORITY,
        "live_source_plan_authority": None,
        "structurally_valid_candidate_count": len(valid),
        "local_evidence_rechecked_candidate_count": (
            len(valid) if locally_rechecked else 0
        ),
        "actual_verified_candidate_count": len(valid) if locally_verified else 0,
        "candidate_deficit": deficit,
        "by_split_family": cells,
        "rejected": sorted(rejected, key=lambda row: row["slot_id"]),
        "eq_policy_mode": eq_mode,
        "v1_contract_unchanged": True,
        "unmodified_level5_challenge_required": True,
    }
    return _seal(result)


def issue_source_manifest_v2_noreplace(
    draft: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    repository_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """실제 local bytes/plant/git ancestry가 모두 PASS일 때만 issued receipt를 발행.

    이 receipt는 acquisition input 자격일 뿐 recorded-v2 live source plan이나 스피커 출력
    authority가 아니다.
    """

    if SOURCE_MANIFEST_V2_ISSUER_AUTHORITY is None:
        raise SourceContractV2Blocked(
            "v2 issuer authority가 닫혀 있어 actual acquisition receipt를 발행할 수 없습니다"
        )
    root = Path(repository_root).resolve(strict=True)
    audit = validate_source_manifest_v2(
        draft,
        campaign=campaign,
        repository_root=root,
        require_local_files=True,
    )
    if audit["status"] != "VERIFIED_48_LOCAL_NOT_LIVE_PLAN":
        raise SourceContractV2Blocked("48개 local v2 candidate가 모두 검증되지 않았습니다")
    output = Path(output_path)
    output = output if output.is_absolute() else root / output
    resolved_parent = output.parent.resolve(strict=True)
    resolved_parent.relative_to(root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"v2 issued target을 덮어쓸 수 없습니다: {output}")
    payload = _seal(
        {
            "schema": BROADBAND_SOURCE_MANIFEST_V2_ISSUED_SCHEMA,
            "role": "verified_acquisition_input_not_live_source_plan",
            "status": "VERIFIED_48_LOCAL_NOT_LIVE_PLAN",
            "draft_evidence_sha256": draft["evidence_sha256"],
            "audit_evidence_sha256": audit["evidence_sha256"],
            "contract_sha256": source_contract_v2()["contract_sha256"],
            "live_authority": None,
        }
    )
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


__all__ = [
    "BOUNDARY_FADE_FRAMES",
    "BROADBAND_SOURCE_CONTRACT_V2_SCHEMA",
    "BROADBAND_SOURCE_MANIFEST_V2_AUDIT_SCHEMA",
    "BROADBAND_SOURCE_MANIFEST_V2_ISSUED_SCHEMA",
    "BROADBAND_SOURCE_MANIFEST_V2_SCHEMA",
    "EQ_POLICY_MODES",
    "MAX_EQ_ATTENUATION_DB",
    "MAX_EQ_BOOST_DB",
    "MAX_EQ_CREST_INCREASE_DB",
    "MIN_SHORT_COMPONENTS",
    "SOURCE_MODES",
    "SourceContractV2Blocked",
    "boundary_fade_coefficients_sha256",
    "issue_source_manifest_v2_noreplace",
    "source_contract_v2",
    "validate_source_manifest_v2",
]
