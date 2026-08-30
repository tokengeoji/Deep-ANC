"""광대역 recorded-v2 소스 acquisition inventory의 fail-closed 계약.

이 모듈은 오디오 장치를 열지 않고, 외부 파일을 복사하지 않는다. 캠페인의 48개
``split x family`` 슬롯과 실제 source evidence 사이의 빈칸을 계산한다. 파일 수나 확장자,
48 kHz로 가공된 WAV만으로는 후보를 세지 않는다. 한 후보를 세려면 native header,
immutable 원본 SHA, decoder/decoded-PCM lineage, 15초 연속 window, 11.314 kHz까지의
실제 spectral evidence, family별 connected-component lineage가 모두 있어야 한다.

현재 정책은 lossless native 원본만 허용한다. 다만 향후 정책 검토가 원본 provenance를
잃지 않도록 lossy 원본도 immutable compressed SHA, decoder fingerprint, decoded PCM SHA를
서로 다른 필드로 기록한다. 이 세 필드가 모두 있어도 ``lossless=false``이면 현재
canonical 후보에는 포함되지 않는다.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..dsp.control_band_contract import ControlBandContract
from .broadband_recording_campaign import (
    BROADBAND_RECORDING_CAMPAIGN_SCHEMA,
    CANDIDATE_SEGMENT_COUNT,
    MIN_NATIVE_SAMPLE_RATE_HZ,
    RECORDING_SECONDS,
    REQUIRED_FAMILIES,
    REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
    REQUIRED_SPLITS,
    validate_source_candidate_metadata,
)


BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA = (
    "broadband_recorded_v2_source_acquisition_manifest_v1"
)
BROADBAND_SOURCE_INVENTORY_AUDIT_SCHEMA = (
    "broadband_recorded_v2_source_inventory_audit_v1"
)
SOURCE_ACQUISITION_MANIFEST_ROLES = {
    "INCOMPLETE": "partial_candidate_evidence_not_live_plan",
    "PASS": "all_slots_verified_but_not_live_plan",
}

_HEX = frozenset("0123456789abcdef")
_REMOTE_RE = re.compile(r"^[A-Za-z0-9_.-]+:[^\r\n]*$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return text


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}가 finite number가 아닙니다")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}가 finite number가 아닙니다") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}가 finite number가 아닙니다")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}가 양의 정수가 아닙니다")
    return value


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("evidence_sha256", None)
    sealed["evidence_sha256"] = _sha256_bytes(_canonical_json(sealed))
    return sealed


def acquisition_manifest_contract() -> dict[str, Any]:
    """사람이 작성할 최소 acquisition manifest의 기계 판독 계약."""

    control = ControlBandContract.broadband_point_control()
    payload: dict[str, Any] = {
        "schema": BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA,
        "minimum_candidate_count": 48,
        "required_splits": list(REQUIRED_SPLITS),
        "required_families": list(REQUIRED_FAMILIES),
        "minimum_groups_per_split_family": 4,
        "minimum_native_sample_rate_hz": MIN_NATIVE_SAMPLE_RATE_HZ,
        "required_native_nyquist_hz": REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
        "minimum_contiguous_untouched_seconds": RECORDING_SECONDS,
        "repeat_or_concatenation_forbidden": True,
        "lossless_native_required": True,
        "immutable_compressed_provenance_fields_retained_for_policy_review": [
            "origin_audio.immutable_source_sha256",
            "decode_provenance.decoder_fingerprint_sha256",
            "decode_provenance.decoded_native_pcm_sha256",
        ],
        "actual_spectral_evidence_required_through_hz": (
            REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
        ),
        "candidate_required_sections": [
            "candidate_metadata",
            "origin_audio",
            "decode_provenance",
            "spectral_evidence",
            "lineage_evidence",
            "corpus_disjointness",
        ],
        "control_band_contract_sha256": control.digest(),
        "live_authority": None,
    }
    payload["contract_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _campaign_from_value(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if value.get("schema") == BROADBAND_RECORDING_CAMPAIGN_SCHEMA:
        return value
    campaign = value.get("campaign")
    if isinstance(campaign, Mapping) and campaign.get("schema") == (
        BROADBAND_RECORDING_CAMPAIGN_SCHEMA
    ):
        return campaign
    raise ValueError("광대역 missing-source campaign을 찾을 수 없습니다")


def required_campaign_slots(value: Mapping[str, Any]) -> list[dict[str, str]]:
    """캠페인의 exact 48개 슬롯과 12-cell 하한을 다시 검산한다."""

    campaign = _campaign_from_value(value)
    slots = campaign.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("광대역 campaign slot이 비었습니다")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for index, raw in enumerate(slots):
        if not isinstance(raw, Mapping):
            raise ValueError(f"campaign slot#{index}가 object가 아닙니다")
        slot_id = str(raw.get("slot_id", "")).strip()
        split = str(raw.get("split", ""))
        family = str(raw.get("source_family", ""))
        if not slot_id or slot_id in seen:
            raise ValueError("campaign slot_id가 비었거나 중복입니다")
        if split not in REQUIRED_SPLITS or family not in REQUIRED_FAMILIES:
            raise ValueError(f"{slot_id}: split/family가 canonical 집합 밖입니다")
        seen.add(slot_id)
        counts[(split, family)] += 1
        result.append({"slot_id": slot_id, "split": split, "source_family": family})
    expected = {
        (split, family): 4
        for split in REQUIRED_SPLITS
        for family in REQUIRED_FAMILIES
    }
    if len(result) != 48 or dict(counts) != expected:
        raise ValueError(
            "campaign은 split×family마다 4개, 총 48개 슬롯이어야 합니다"
        )
    return result


def _matrix(value: object, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != CANDIDATE_SEGMENT_COUNT:
        raise ValueError(f"{label}는 9개 segment여야 합니다")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError(f"{label}의 각 segment는 7대역이어야 합니다")
        result.append(
            [_finite_float(item, label=f"{label} value") for item in row]
        )
    return result


def _candidate_reasons(
    raw: object,
    *,
    slot: Mapping[str, str],
    control: ControlBandContract,
) -> tuple[list[str], dict[str, str] | None]:
    reasons: list[str] = []
    try:
        value = _exact_keys(
            raw,
            {
                "slot_id",
                "candidate_metadata",
                "origin_audio",
                "decode_provenance",
                "spectral_evidence",
                "lineage_evidence",
                "corpus_disjointness",
            },
            label=f"{slot['slot_id']} acquisition candidate",
        )
    except ValueError as exc:
        return [str(exc)], None
    if value["slot_id"] != slot["slot_id"]:
        reasons.append("slot_id_mismatch")

    metadata = value["candidate_metadata"]
    metadata_result: dict[str, Any] | None = None
    try:
        metadata_result = validate_source_candidate_metadata(
            metadata,
            expected_split=slot["split"],
            expected_family=slot["source_family"],
        )
    except (TypeError, ValueError) as exc:
        reasons.append(f"candidate_metadata_invalid:{exc}")

    try:
        origin = _exact_keys(
            value["origin_audio"],
            {
                "storage_locator",
                "size_bytes",
                "immutable_source_sha256",
                "container",
                "codec",
                "subtype",
                "lossless",
                "native_sample_rate_hz",
                "native_nyquist_hz",
                "channels",
                "frame_count",
                "duration_seconds",
                "contiguous_window_start_frame",
                "contiguous_window_frames",
                "header_evidence_sha256",
            },
            label=f"{slot['slot_id']}.origin_audio",
        )
        locator = str(origin["storage_locator"])
        if not locator.strip() or "\n" in locator or "\r" in locator:
            reasons.append("storage_locator_invalid")
        _positive_int(origin["size_bytes"], label="origin size")
        immutable_sha = _sha(
            origin["immutable_source_sha256"], label="immutable source SHA"
        )
        _sha(origin["header_evidence_sha256"], label="header evidence SHA")
        if not all(str(origin[key]).strip() for key in ("container", "codec", "subtype")):
            reasons.append("native_container_codec_subtype_missing")
        native_rate = _positive_int(
            origin["native_sample_rate_hz"], label="native sample rate"
        )
        nyquist = _finite_float(origin["native_nyquist_hz"], label="native Nyquist")
        if not math.isclose(nyquist, native_rate / 2.0, rel_tol=0.0, abs_tol=1e-9):
            reasons.append("native_rate_nyquist_mismatch")
        if native_rate < MIN_NATIVE_SAMPLE_RATE_HZ:
            reasons.append("native_rate_below_22628_hz")
        if nyquist < REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ:
            reasons.append("native_nyquist_below_11313_708_hz")
        channels = _positive_int(origin["channels"], label="native channels")
        if channels != 1:
            reasons.append("native_source_not_mono")
        frames = _positive_int(origin["frame_count"], label="native frame count")
        duration = _finite_float(origin["duration_seconds"], label="native duration")
        if not math.isclose(duration, frames / native_rate, rel_tol=0.0, abs_tol=1e-6):
            reasons.append("native_duration_frame_count_mismatch")
        window_start = origin["contiguous_window_start_frame"]
        window_frames = origin["contiguous_window_frames"]
        if (
            isinstance(window_start, bool)
            or not isinstance(window_start, int)
            or window_start < 0
            or isinstance(window_frames, bool)
            or not isinstance(window_frames, int)
        ):
            reasons.append("contiguous_window_frame_contract_invalid")
        else:
            required_native_frames = math.ceil(RECORDING_SECONDS * native_rate)
            if window_frames < required_native_frames or window_start + window_frames > frames:
                reasons.append("contiguous_native_window_shorter_than_15s")
        if origin["lossless"] is not True:
            reasons.append("native_source_not_lossless_current_policy")
        if isinstance(metadata, Mapping):
            if metadata.get("native_sample_rate_hz") != native_rate:
                reasons.append("origin_metadata_native_rate_mismatch")
            if metadata.get("native_content_sha256") != immutable_sha:
                reasons.append("origin_metadata_native_sha_mismatch")
            if metadata.get("lossless") is not origin["lossless"]:
                reasons.append("origin_metadata_lossless_mismatch")
    except (TypeError, ValueError) as exc:
        reasons.append(f"origin_audio_invalid:{exc}")
        origin = None
        immutable_sha = ""
        native_rate = 0
        frames = 0
        window_start = 0
        window_frames = 0

    try:
        decode = _exact_keys(
            value["decode_provenance"],
            {
                "decoder_fingerprint_sha256",
                "decoder_receipt_sha256",
                "decoded_native_pcm_sha256",
                "decoded_sample_rate_hz",
                "decoded_channels",
                "decoded_frames",
            },
            label=f"{slot['slot_id']}.decode_provenance",
        )
        _sha(decode["decoder_fingerprint_sha256"], label="decoder fingerprint SHA")
        _sha(decode["decoder_receipt_sha256"], label="decoder receipt SHA")
        decoded_sha = _sha(
            decode["decoded_native_pcm_sha256"], label="decoded native PCM SHA"
        )
        if decode["decoded_sample_rate_hz"] != native_rate:
            reasons.append("decoded_native_rate_mismatch")
        if decode["decoded_channels"] != 1:
            reasons.append("decoded_native_channels_not_mono")
        if decode["decoded_frames"] != frames:
            reasons.append("decoded_native_frame_count_mismatch")
    except (TypeError, ValueError) as exc:
        reasons.append(f"decode_provenance_invalid:{exc}")
        decoded_sha = ""

    try:
        spectral = _exact_keys(
            value["spectral_evidence"],
            {
                "evidence_sha256",
                "decoded_native_pcm_sha256",
                "control_band_contract_sha256",
                "canonical_fullband_plant_evidence_sha256",
                "window_start_frame",
                "window_frames",
                "analysed_upper_hz",
                "actual_11314_hz_covered",
                "source_density_ratios",
                "predicted_err_density_ratios",
            },
            label=f"{slot['slot_id']}.spectral_evidence",
        )
        _sha(spectral["evidence_sha256"], label="spectral evidence SHA")
        if _sha(
            spectral["decoded_native_pcm_sha256"], label="spectral decoded PCM SHA"
        ) != decoded_sha:
            reasons.append("spectral_decoded_pcm_sha_mismatch")
        if spectral["control_band_contract_sha256"] != control.digest():
            reasons.append("spectral_control_band_contract_mismatch")
        _sha(
            spectral["canonical_fullband_plant_evidence_sha256"],
            label="spectral canonical plant SHA",
        )
        if (
            spectral["window_start_frame"] != window_start
            or spectral["window_frames"] != window_frames
        ):
            reasons.append("spectral_contiguous_window_mismatch")
        upper = _finite_float(spectral["analysed_upper_hz"], label="spectral upper")
        if (
            upper < REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
            or spectral["actual_11314_hz_covered"] is not True
        ):
            reasons.append("actual_11314_hz_spectral_evidence_absent")
        source_matrix = _matrix(
            spectral["source_density_ratios"], label="spectral source density"
        )
        predicted_matrix = _matrix(
            spectral["predicted_err_density_ratios"],
            label="spectral predicted ERR density",
        )
        if isinstance(metadata, Mapping):
            if metadata.get("source_density_ratios") != source_matrix:
                reasons.append("spectral_source_density_metadata_mismatch")
            if metadata.get("predicted_err_density_ratios") != predicted_matrix:
                reasons.append("spectral_predicted_err_density_metadata_mismatch")
    except (TypeError, ValueError) as exc:
        reasons.append(f"spectral_evidence_invalid:{exc}")

    try:
        lineage = _exact_keys(
            value["lineage_evidence"],
            {
                "component_id",
                "authority_sha256",
                "family_values_sha256",
                "connected_component_keys",
                "dsu_verified",
            },
            label=f"{slot['slot_id']}.lineage_evidence",
        )
        component_id = str(lineage["component_id"]).strip()
        if not component_id:
            reasons.append("lineage_component_id_empty")
        _sha(lineage["authority_sha256"], label="lineage authority SHA")
        _sha(lineage["family_values_sha256"], label="lineage family values SHA")
        keys = lineage["connected_component_keys"]
        expected_keys = set(
            # source_candidate_requirements를 통하지 않고 현재 campaign row의 exact
            # requirement를 소비해 family-specific key drift를 막는다.
            _campaign_lineage_keys(slot["source_family"])
        )
        if not isinstance(keys, list) or set(map(str, keys)) != expected_keys:
            reasons.append("lineage_connected_component_keys_mismatch")
        if lineage["dsu_verified"] is not True:
            reasons.append("lineage_dsu_not_verified")
        if isinstance(metadata, Mapping) and metadata.get("lineage_component_id") != component_id:
            reasons.append("lineage_metadata_component_mismatch")
    except (TypeError, ValueError) as exc:
        reasons.append(f"lineage_evidence_invalid:{exc}")
        component_id = ""

    try:
        disjoint = _exact_keys(
            value["corpus_disjointness"],
            {
                "authority_sha256",
                "raw_content_disjoint",
                "processed_content_disjoint",
                "lineage_component_disjoint",
            },
            label=f"{slot['slot_id']}.corpus_disjointness",
        )
        _sha(disjoint["authority_sha256"], label="corpus disjointness authority SHA")
        if any(
            disjoint[key] is not True
            for key in (
                "raw_content_disjoint",
                "processed_content_disjoint",
                "lineage_component_disjoint",
            )
        ):
            reasons.append("recorded_synthetic_corpus_disjointness_not_verified")
    except (TypeError, ValueError) as exc:
        reasons.append(f"corpus_disjointness_invalid:{exc}")

    identity = None
    if metadata_result is not None:
        identity = {
            "native_content_sha256": str(metadata_result["native_content_sha256"]),
            "processed_content_sha256": str(
                metadata_result["processed_content_sha256"]
            ),
            "transform_receipt_sha256": str(
                metadata_result["transform_receipt_sha256"]
            ),
            "decoded_native_pcm_sha256": decoded_sha,
            "lineage_component_id": component_id,
        }
    return sorted(set(reasons)), identity


def _campaign_lineage_keys(family: str) -> tuple[str, ...]:
    if family == "speech":
        return ("raw_content", "speaker", "book_or_work", "recording_session")
    if family == "music":
        return (
            "raw_content",
            "perceptual_audio_alias",
            "artist",
            "album_or_release",
            "recording_session",
        )
    if family == "environment":
        return (
            "raw_content",
            "original_recording_id",
            "field_recording_session",
            "captured_event",
        )
    if family == "machine":
        return (
            "raw_content",
            "physical_machine_unit",
            "operating_run",
            "recording_session",
        )
    raise ValueError(f"알 수 없는 source family: {family}")


def audit_acquisition_manifest(
    campaign_value: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """검증된 후보만 세고 12-cell의 정확한 부족분을 반환한다."""

    slots = required_campaign_slots(campaign_value)
    control = ControlBandContract.broadband_point_control()
    slot_by_id = {row["slot_id"]: row for row in slots}
    candidates: list[Any] = []
    manifest_declared_status = None
    manifest_evidence = None
    manifest_role = None
    top_level_reasons: list[str] = []
    if manifest is not None:
        try:
            value = _exact_keys(
                manifest,
                {
                    "schema",
                    "role",
                    "status",
                    "contract_sha256",
                    "control_band_contract_sha256",
                    "campaign_evidence_sha256",
                    "candidates",
                    "evidence_sha256",
                },
                label="source acquisition manifest",
            )
            if value["schema"] != BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA:
                top_level_reasons.append("manifest_schema_mismatch")
            contract = acquisition_manifest_contract()
            if value["contract_sha256"] != contract["contract_sha256"]:
                top_level_reasons.append("manifest_contract_sha_mismatch")
            if value["control_band_contract_sha256"] != control.digest():
                top_level_reasons.append("manifest_control_band_sha_mismatch")
            campaign = _campaign_from_value(campaign_value)
            if value["campaign_evidence_sha256"] != campaign.get("evidence_sha256"):
                top_level_reasons.append("manifest_campaign_evidence_sha_mismatch")
            manifest_declared_status = str(value["status"])
            manifest_role = str(value["role"])
            if manifest_declared_status not in SOURCE_ACQUISITION_MANIFEST_ROLES:
                top_level_reasons.append("manifest_declared_status_invalid")
            elif manifest_role != SOURCE_ACQUISITION_MANIFEST_ROLES[manifest_declared_status]:
                top_level_reasons.append("manifest_role_status_mismatch")
            candidates = value["candidates"]
            if not isinstance(candidates, list):
                top_level_reasons.append("manifest_candidates_not_list")
                candidates = []
            stored_evidence = _sha(
                value["evidence_sha256"], label="manifest evidence SHA"
            )
            expected_evidence = _sha256_bytes(
                _canonical_json(
                    {key: item for key, item in value.items() if key != "evidence_sha256"}
                )
            )
            if stored_evidence != expected_evidence:
                top_level_reasons.append("manifest_evidence_sha_mismatch")
            manifest_evidence = stored_evidence
        except (TypeError, ValueError) as exc:
            top_level_reasons.append(f"manifest_top_level_invalid:{exc}")

    results: dict[str, dict[str, Any]] = {
        slot_id: {
            "slot_id": slot_id,
            "split": slot["split"],
            "source_family": slot["source_family"],
            "status": "MISSING",
            "reasons": ["candidate_evidence_missing"],
            "identity": None,
        }
        for slot_id, slot in slot_by_id.items()
    }
    unknown_or_duplicate: list[str] = []
    for index, raw in enumerate(candidates):
        slot_id = str(raw.get("slot_id", "")) if isinstance(raw, Mapping) else ""
        if slot_id not in slot_by_id or results[slot_id]["status"] != "MISSING":
            unknown_or_duplicate.append(slot_id or f"candidate#{index}")
            continue
        reasons, identity = _candidate_reasons(
            raw, slot=slot_by_id[slot_id], control=control
        )
        results[slot_id] = {
            "slot_id": slot_id,
            "split": slot_by_id[slot_id]["split"],
            "source_family": slot_by_id[slot_id]["source_family"],
            "status": "PASS" if not reasons else "REJECTED",
            "reasons": reasons,
            "identity": identity,
        }
    if unknown_or_duplicate:
        top_level_reasons.append("unknown_or_duplicate_candidate_slots")

    # 동일 bytes/decoded PCM/transform/lineage를 여러 독립 group으로 위장하면 관련된
    # 모든 행을 탈락시킨다. 입력 순서에 따라 첫 행만 살아남는 정책을 쓰지 않는다.
    identity_slots: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    identity_fields = (
        "native_content_sha256",
        "processed_content_sha256",
        "transform_receipt_sha256",
        "decoded_native_pcm_sha256",
        "lineage_component_id",
    )
    for slot_id, row in results.items():
        identity = row["identity"]
        if row["status"] == "PASS" and isinstance(identity, Mapping):
            for field in identity_fields:
                identity_slots[(field, str(identity[field]))].append(slot_id)
    for (field, _), duplicates in identity_slots.items():
        if len(duplicates) <= 1:
            continue
        for slot_id in duplicates:
            results[slot_id]["status"] = "REJECTED"
            results[slot_id]["reasons"] = sorted(
                set(results[slot_id]["reasons"] + [f"duplicate_{field}"])
            )

    counts: Counter[tuple[str, str]] = Counter()
    for row in results.values():
        if row["status"] == "PASS":
            counts[(row["split"], row["source_family"])] += 1
    cells = []
    total_deficit = 0
    for split in REQUIRED_SPLITS:
        for family in REQUIRED_FAMILIES:
            eligible = counts[(split, family)]
            deficit = max(0, 4 - eligible)
            total_deficit += deficit
            cells.append(
                {
                    "split": split,
                    "source_family": family,
                    "required_independent_groups": 4,
                    "eligible_independent_groups": eligible,
                    "deficit": deficit,
                }
            )
    eligible_count = sum(counts.values())
    recomputed_status = "PASS" if not top_level_reasons and total_deficit == 0 else "BLOCKED"
    if manifest_declared_status == "PASS" and recomputed_status != "PASS":
        top_level_reasons.append("declared_PASS_but_recomputed_BLOCKED")
    if manifest_declared_status == "INCOMPLETE" and recomputed_status == "PASS":
        top_level_reasons.append("declared_INCOMPLETE_but_recomputed_PASS")
        recomputed_status = "BLOCKED"

    public_rows = []
    for row in results.values():
        public_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "identity"
            }
        )
    payload = {
        "schema": BROADBAND_SOURCE_INVENTORY_AUDIT_SCHEMA,
        "status": recomputed_status,
        "control_band_contract_sha256": control.digest(),
        "manifest_present": manifest is not None,
        "manifest_declared_status": manifest_declared_status,
        "manifest_role": manifest_role,
        "manifest_evidence_sha256": manifest_evidence,
        "top_level_reasons": sorted(set(top_level_reasons)),
        "unknown_or_duplicate_candidate_slots": sorted(unknown_or_duplicate),
        "required_candidate_count": 48,
        "eligible_candidate_count": eligible_count,
        "candidate_deficit": total_deficit,
        "by_split_family": cells,
        "candidate_results": public_rows,
    }
    return _seal(payload)


def _listing_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for row in rows:
        hashes = row.get("Hashes")
        normalized.append(
            {
                "path": str(row.get("Path", "")),
                "size": int(row.get("Size", 0)),
                "mod_time": str(row.get("ModTime", "")),
                "hashes": dict(sorted(hashes.items())) if isinstance(hashes, dict) else {},
            }
        )
    return _sha256_bytes(_canonical_json(sorted(normalized, key=lambda item: item["path"])))


def collect_rclone_metadata_summary(
    remote_root: str,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Google Drive 등을 ``rclone lsjson`` 목록/메타데이터로만 감사한다.

    파일 content를 열거나 복사하지 않는다. 보고서에는 개별 path/hash를 넣지 않고 전체
    listing digest와 집계만 남긴다. 원격 metadata만으로 native header/길이/lineage/PSD를
    증명할 수 없으므로 eligible count는 항상 0이다.
    """

    if not _REMOTE_RE.fullmatch(remote_root):
        raise ValueError("rclone remote root 형식이 안전하지 않습니다")
    root = remote_root.rstrip("/")
    cohorts = {
        "raw_speech": "raw/speech",
        "raw_music": "raw/music",
        "raw_noise": "raw/noise",
        "processed_source_pool_v1": "source_pool",
        "processed_source_pool_v2": "source_pool_v2",
    }
    summaries: dict[str, Any] = {}
    for label, suffix in cohorts.items():
        target = f"{root}/{suffix}"
        command = [
            "rclone",
            "lsjson",
            "--recursive",
            "--files-only",
            "--hash",
            target,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            summaries[label] = {
                "status": "BLOCKED_REMOTE_METADATA_UNAVAILABLE",
                "file_count": 0,
                "byte_count": 0,
                "extension_counts": {},
                "listing_evidence_sha256": None,
                "eligible_candidate_count": 0,
                "reason": "rclone_lsjson_nonzero_without_stderr_disclosure",
            }
            continue
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError:
            summaries[label] = {
                "status": "BLOCKED_REMOTE_METADATA_INVALID",
                "file_count": 0,
                "byte_count": 0,
                "extension_counts": {},
                "listing_evidence_sha256": None,
                "eligible_candidate_count": 0,
                "reason": "rclone_lsjson_invalid_json",
            }
            continue
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("rclone lsjson 최상위가 file object list가 아닙니다")
        extensions = Counter(
            Path(str(row.get("Path", ""))).suffix.lower() or "<none>"
            for row in rows
        )
        summaries[label] = {
            "status": "PASS_READ_ONLY_METADATA_LISTING_ONLY",
            "file_count": len(rows),
            "byte_count": sum(int(row.get("Size", 0)) for row in rows),
            "extension_counts": dict(sorted(extensions.items())),
            "listing_evidence_sha256": _listing_digest(rows),
            "eligible_candidate_count": 0,
            "reason": (
                "remote_object_metadata_cannot_prove_native_header_contiguous_15s_"
                "lineage_decoder_pcm_and_actual_11314hz_spectrum"
            ),
        }
    return {
        "access": "read_only_rclone_lsjson",
        "external_content_read_or_copied": False,
        "remote_root": root,
        "cohorts": summaries,
    }


def collect_local_source_summary(repository_root: str | Path) -> dict[str, Any]:
    """현 로컬 cohort의 header/CSV만 읽고 canonical 후보로 자동 승격하지 않는다."""

    import csv
    import soundfile as sf

    root = Path(repository_root).resolve(strict=True)
    cohort_specs = {
        "raw_speech_flac": list((root / "data/raw/speech").rglob("*.flac")),
        "raw_environment_esc50_wav": list(
            (root / "data/raw/noise/esc50").rglob("*.wav")
        ),
        "processed_source_pool_v1_wav": list(
            (root / "data/source_pool").rglob("*.wav")
        ),
        "processed_source_pool_v2_wav": list(
            (root / "data/source_pool_v2").rglob("*.wav")
        ),
    }
    cohorts: dict[str, Any] = {}
    for label, paths in cohort_specs.items():
        rows = []
        for path in sorted(paths):
            info = sf.info(str(path))
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "format": str(info.format or ""),
                    "subtype": str(info.subtype or ""),
                }
            )
        rate_counts = Counter(row["sample_rate"] for row in rows)
        durations = [row["frames"] / row["sample_rate"] for row in rows]
        cohorts[label] = {
            "file_count": len(rows),
            "sample_rate_counts": {
                str(key): value for key, value in sorted(rate_counts.items())
            },
            "minimum_duration_seconds": min(durations) if durations else None,
            "maximum_duration_seconds": max(durations) if durations else None,
            "header_listing_evidence_sha256": _sha256_bytes(_canonical_json(rows)),
            "eligible_candidate_count": 0,
        }
    cohorts["raw_speech_flac"]["reason"] = (
        "native_16khz_below_22628hz_and_11313_708hz_nyquist_requirement"
    )
    cohorts["raw_environment_esc50_wav"]["reason"] = (
        "five_second_original_has_no_contiguous_untouched_15s_window;_repeat_or_concat_forbidden"
    )
    for label in ("processed_source_pool_v1_wav", "processed_source_pool_v2_wav"):
        cohorts[label]["reason"] = (
            "processed_48khz_pool_lacks_native_raw_transform_decoder_lineage_and_11314hz_evidence"
        )

    csv_summaries = []
    for relative in ("data/source_pool/sources.csv", "data/source_pool_v2/sources.csv"):
        path = root / relative
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        csv_summaries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": len(rows),
                "fields": fields,
                "native_provenance_fields_present": all(
                    field in fields
                    for field in (
                        "native_sample_rate_hz",
                        "native_content_sha256",
                        "transform_receipt_sha256",
                    )
                ),
            }
        )
    return {
        "access": "local_header_and_csv_read_only",
        "files_modified": False,
        "cohorts": cohorts,
        "source_pool_csvs": csv_summaries,
    }


def inspect_existing_pipeline_capability(repository_root: str | Path) -> dict[str, Any]:
    """현재 script가 placeholder 생성/READY plan 검증/plan 발행 중 무엇을 하는지 감사."""

    root = Path(repository_root).resolve(strict=True)
    files = {
        "campaign_auditor": root / "scripts/data/audit_broadband_recording_campaign.py",
        "recorded_v2_cli": root / "scripts/data/record_broadband_v2.py",
        "campaign_contract": root / "src/deep_anc/data/broadband_recording_campaign.py",
        "capture_contract": root / "src/deep_anc/data/recorded_v2_capture.py",
    }
    evidence = {}
    symbols: dict[str, set[str]] = {}
    for label, path in files.items():
        if not path.is_file():
            evidence[label] = {"path": path.relative_to(root).as_posix(), "exists": False}
            symbols[label] = set()
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        symbols[label] = names | attrs
        evidence[label] = {
            "path": path.relative_to(root).as_posix(),
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    default_plan = root / "data/source_plans/recorded_broadband_v2/canonical_v1.json"
    campaign_builder = "build_missing_source_campaign" in symbols["campaign_auditor"]
    source_validator = "validate_source_plan" in symbols["recorded_v2_cli"]
    known_publish_symbols = {
        "publish_source_plan",
        "build_source_plan",
        "write_source_plan",
        "issue_source_plan",
    }
    publisher = bool(symbols["recorded_v2_cli"] & known_publish_symbols)
    return {
        "evidence": evidence,
        "placeholder_campaign_builder_present": campaign_builder,
        "ready_source_plan_validator_present": source_validator,
        "ready_source_plan_publisher_present": publisher,
        "default_ready_source_plan_path": default_plan.relative_to(root).as_posix(),
        "default_ready_source_plan_exists": default_plan.is_file(),
        "can_currently_make_actual_48_source_live_plan": bool(
            campaign_builder and source_validator and publisher and default_plan.is_file()
        ),
        "verdict": (
            "BLOCKED_PLACEHOLDER_AND_VALIDATOR_EXIST_BUT_VERIFIED_ACQUISITION_"
            "MANIFEST_AND_SOURCE_PLAN_PUBLISHER_ARE_ABSENT"
        ),
    }


__all__ = [
    "BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA",
    "BROADBAND_SOURCE_INVENTORY_AUDIT_SCHEMA",
    "SOURCE_ACQUISITION_MANIFEST_ROLES",
    "acquisition_manifest_contract",
    "audit_acquisition_manifest",
    "collect_local_source_summary",
    "collect_rclone_metadata_summary",
    "inspect_existing_pipeline_capability",
    "required_campaign_slots",
]
