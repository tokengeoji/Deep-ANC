"""광대역 recorded-v2 신규 소스 캠페인의 최소 하한과 후보 사전검사.

이 모듈은 오디오 장치를 열거나 파일을 내려받지 않는다. 기존 82세션 진단에서
``split x family x 7 subband``별 독립 group 부족분을 다시 계산하고, 아직 확보하지
않은 신규 원본이 만족해야 할 조건만 만든다. 슬롯 이름은 실제 source/group을 뜻하지
않으며, 실제 원본 path/SHA/lineage를 채워 넣기 전 상태는 항상 ``BLOCKED``다.

광대역 캠페인은 저역과 고역을 같은 segment에서 함께 학습시키는 것이 목적이다. 따라서
각 신규 group은 일곱 대역을 모두 통과해야 하며, 한 group이 일부 대역만 채우는 경우의
더 큰 캠페인 수를 최소 하한으로 오인하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .broadband_coverage_receipt import (
    MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND,
    MIN_JOINT_SEGMENTS_PER_GROUP,
    MIN_SOURCE_ERR_COHERENCE,
    MIN_TARGET_D_DENSITY_RATIO,
    NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE,
    NATIVE_EXACT_TARGET_RATE_ROLE,
    minimum_native_sample_rate_hz,
)
from ..dsp.control_band_contract import ControlBandContract


BROADBAND_RECORDING_CAMPAIGN_SCHEMA = "broadband_recorded_v2_missing_source_campaign_v2"
BROADBAND_DIAGNOSTIC_SCHEMA = "broadband_prerequisite_audit_v1"
BROADBAND_COVERAGE_DIAGNOSTIC_SCHEMA = "recorded_broadband_coverage_diagnostic_v1"

# native coverage와 48 kHz DAC processing rate를 분리한다. native Nyquist가 8 kHz
# 중심 octave 상단(11.314 kHz)을 실제로 포함하면 1회 polyphase rate conversion을 허용한다.
# resample 뒤 생긴 bin을 새 bandwidth로 주장하는 것은 계속 금지한다.
_BROADBAND_CONTRACT = ControlBandContract.broadband_point_control()
MIN_NATIVE_SAMPLE_RATE_HZ = minimum_native_sample_rate_hz(_BROADBAND_CONTRACT)
REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ = _BROADBAND_CONTRACT.required_excitation_upper_hz

# Stage-1에서 검증한 15초/1.5초/0.25초 고정 population을 광대역 소스 선택에도 사용한다.
# 9개 비중첩 segment 중 8개가 일곱 대역을 동시에 통과해야 한다.
RECORDING_SECONDS = 15.0
SEGMENT_SECONDS = 1.5
SEGMENT_START_OFFSET_SECONDS = 0.25
CANDIDATE_SEGMENT_COUNT = 9

# 고정 안전 peak에서 지나치게 작은 RMS가 되어 고역 coherence가 사라지는 후보를 소리를
# 내기 전에 제거하는 acquisition ceiling이다. 동적 압축/클리핑으로 이 값을 맞추는 것은
# 금지하고, 원본의 다른 untouched window 또는 다른 원본을 선택한다.
MAX_SOURCE_CREST_FACTOR_DB = 15.0

REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_FAMILIES = ("speech", "music", "environment", "machine")

_PROCESSING_BY_NATIVE_RATE = {
    NATIVE_EXACT_TARGET_RATE_ROLE: lambda rate: rate == 48_000,
    NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE: lambda rate: (
        rate >= MIN_NATIVE_SAMPLE_RATE_HZ and rate != 48_000
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _metadata_sha(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} SHA-256이 유효하지 않습니다")
    return digest


def _coverage_summary(diagnostic: Mapping[str, Any]) -> tuple[ControlBandContract, Mapping[str, Any]]:
    if diagnostic.get("schema") != BROADBAND_DIAGNOSTIC_SCHEMA:
        raise ValueError("광대역 prerequisite diagnostic schema가 다릅니다")
    coverage = diagnostic.get("recorded_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("82세션 recorded coverage가 없습니다")
    if coverage.get("schema") != BROADBAND_COVERAGE_DIAGNOSTIC_SCHEMA:
        raise ValueError("recorded broadband diagnostic schema가 다릅니다")
    if coverage.get("role") != "diagnostic_only_not_campaign_receipt":
        raise ValueError("diagnostic을 campaign receipt로 승격할 수 없습니다")
    contract = ControlBandContract.broadband_point_control()
    if coverage.get("control_band_contract_sha256") != contract.digest():
        raise ValueError("recorded diagnostic의 control-band SHA가 현재 계약과 다릅니다")
    summary = coverage.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(
        summary.get("by_split_family"), Mapping
    ):
        raise ValueError("recorded diagnostic summary가 없습니다")
    return contract, summary["by_split_family"]


def calculate_minimum_new_groups(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """현재 진단 group 수에서 신규 all-band group의 엄밀한 하한을 계산한다."""

    contract, by_split = _coverage_summary(diagnostic)
    rows: list[dict[str, Any]] = []
    total = 0
    for split in REQUIRED_SPLITS:
        split_rows = by_split.get(split)
        if not isinstance(split_rows, Mapping):
            raise ValueError(f"recorded diagnostic에 {split} split이 없습니다")
        for family in REQUIRED_FAMILIES:
            family_row = split_rows.get(family)
            if not isinstance(family_row, Mapping):
                raise ValueError(f"recorded diagnostic에 {split}/{family}가 없습니다")
            bands = family_row.get("subbands")
            if not isinstance(bands, list) or len(bands) != len(
                contract.point_control_subbands_hz
            ):
                raise ValueError(f"{split}/{family}의 subband 수가 다릅니다")
            observed: list[int] = []
            deficits: list[int] = []
            for index, expected_band in enumerate(contract.point_control_subbands_hz):
                band = bands[index]
                if not isinstance(band, Mapping):
                    raise ValueError(f"{split}/{family} band row가 mapping이 아닙니다")
                value = band.get("joint_pass_independent_groups")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{split}/{family} group count가 유효하지 않습니다")
                observed.append(value)
                deficits.append(
                    max(0, MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND - value)
                )
            # 신규 group 하나가 일곱 대역을 모두 통과하도록 요구하므로 대역별 부족분의
            # 합이 아니라 최댓값이 이 cell의 최소 신규 group 수다.
            minimum = max(deficits)
            total += minimum
            rows.append(
                {
                    "split": split,
                    "source_family": family,
                    "bands_hz": [list(band) for band in contract.point_control_subbands_hz],
                    "observed_joint_groups": observed,
                    "deficit_by_band": deficits,
                    "minimum_new_all_band_groups": minimum,
                }
            )
    return {
        "control_band_contract_sha256": contract.digest(),
        "minimum_groups_per_split_family_band": (
            MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND
        ),
        "rows": rows,
        "minimum_new_source_groups": total,
        "minimum_new_recording_sessions": total,
        "minimum_audible_seconds": total * RECORDING_SECONDS,
    }


def _lineage_requirement(family: str) -> dict[str, Any]:
    if family == "speech":
        component_keys = ["raw_content", "speaker", "book_or_work", "recording_session"]
    elif family == "music":
        component_keys = [
            "raw_content",
            "perceptual_audio_alias",
            "artist",
            "album_or_release",
            "recording_session",
        ]
    elif family == "environment":
        component_keys = [
            "raw_content",
            "original_recording_id",
            "field_recording_session",
            "captured_event",
        ]
    elif family == "machine":
        component_keys = [
            "raw_content",
            "physical_machine_unit",
            "operating_run",
            "recording_session",
        ]
    else:  # pragma: no cover - caller only uses the fixed family set
        raise ValueError(f"알 수 없는 source family: {family}")
    return {
        "connected_component_keys": component_keys,
        "same_component_cannot_cross_slots_or_splits": True,
        "same_raw_or_excerpt_or_augmented_alias_cannot_form_new_group": True,
        "must_be_disjoint_from_existing_recorded_and_all_synthetic_splits": True,
    }


def source_candidate_requirements(family: str) -> dict[str, Any]:
    """아직 존재하지 않는 한 신규 source가 만족해야 할 exact 사전조건."""

    if family not in REQUIRED_FAMILIES:
        raise ValueError(f"source family가 canonical 집합 밖입니다: {family}")
    starts = [
        SEGMENT_START_OFFSET_SECONDS + index * SEGMENT_SECONDS
        for index in range(CANDIDATE_SEGMENT_COUNT)
    ]
    return {
        "source_family": family,
        "native_audio": {
            "minimum_sample_rate_hz": MIN_NATIVE_SAMPLE_RATE_HZ,
            "minimum_native_nyquist_hz": REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
            "processed_playback_sample_rate_hz": 48_000,
            "allowed_processing_roles": list(_PROCESSING_BY_NATIVE_RATE),
            "lossless_pcm_or_flac_required": True,
            "single_polyphase_rate_conversion_if_not_48k_required": True,
            "resampler_verified_passband_upper_hz_minimum": (
                REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
            ),
            "raw_native_processed_and_transform_sha_required": True,
            "resampling_cannot_create_or_claim_native_bandwidth": True,
            "lossy_transcode_for_coverage_forbidden": True,
            "repeat_or_concatenation_for_coverage_forbidden": True,
            "spectral_eq_dynamic_compression_and_clipping_for_coverage_forbidden": True,
            "static_gain_only": True,
        },
        "window": {
            "recording_seconds": RECORDING_SECONDS,
            "minimum_source_window_seconds": RECORDING_SECONDS,
            "segment_seconds": SEGMENT_SECONDS,
            "segment_start_seconds": starts,
            "candidate_segment_count": CANDIDATE_SEGMENT_COUNT,
            "minimum_all_seven_band_pass_segments": MIN_JOINT_SEGMENTS_PER_GROUP,
        },
        "spectrum": {
            "density_definition": (
                "band mean PSD / 150-11313.708Hz union mean PSD"
            ),
            "source_density_minimum": MIN_TARGET_D_DENSITY_RATIO,
            "canonical_P_predicted_ERR_density_minimum": MIN_TARGET_D_DENSITY_RATIO,
            "actual_ERR_target_d_density_minimum": MIN_TARGET_D_DENSITY_RATIO,
            "actual_source_ERR_coherence_minimum": MIN_SOURCE_ERR_COHERENCE,
            "all_seven_bands_same_segment_required": True,
            "canonical_fullband_P_evidence_required_before_selection": True,
        },
        "crest": {
            "maximum_db": MAX_SOURCE_CREST_FACTOR_DB,
            "measured_on_exact_submitted_pcm": True,
            "must_pass_without_dynamic_processing": True,
        },
        "lineage": _lineage_requirement(family),
    }


def build_missing_source_campaign(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    """실제 source를 발명하지 않는 BLOCKED 슬롯 계획을 만든다."""

    minimum = calculate_minimum_new_groups(diagnostic)
    slots: list[dict[str, Any]] = []
    for row in minimum["rows"]:
        for index in range(int(row["minimum_new_all_band_groups"])):
            family = str(row["source_family"])
            split = str(row["split"])
            slots.append(
                {
                    "slot_id": f"{split}-{family}-{index + 1:02d}",
                    "split": split,
                    "source_family": family,
                    "source": None,
                    "lineage_component": None,
                    "requirements": source_candidate_requirements(family),
                }
            )
    payload: dict[str, Any] = {
        "schema": BROADBAND_RECORDING_CAMPAIGN_SCHEMA,
        "role": "missing_source_specification_not_live_plan",
        "status": "BLOCKED_MISSING_VERIFIED_SOURCES_AND_FULLBAND_P",
        "control_band_contract_sha256": minimum["control_band_contract_sha256"],
        "minimum": minimum,
        "slots": slots,
        "live_preconditions": [
            "canonical fullband causal P/S evidence PASS",
            "48개 source raw/SHA/native-rate/lineage/PSD/crest 후보 receipt PASS",
            "recorded-v2 electrical-loopback 또는 pilot fractional-warp dry-run PASS",
            "exact source plan 및 no-replace output path PASS",
            "전체 pytest, 장치 무점유, CPU/clock/input-only preflight PASS",
            "사용자 입회, 볼륨 최저, 배선/기하 확인",
        ],
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_source_candidate_metadata(
    candidate: Mapping[str, Any],
    *,
    expected_split: str,
    expected_family: str,
) -> dict[str, Any]:
    """향후 실제 후보 metadata가 슬롯의 최소 source 조건을 만족하는지 검사한다.

    파일 bytes와 lineage DSU 자체는 후속 publisher가 다시 검사해야 한다. 이 함수는 native
    Nyquist와 48 kHz DAC rate를 혼동하거나, 변환 1회/passband receipt 없이 rate-convert한
    후보, 일부 대역/segment만 통과한 후보를 source plan에 넣기 전에 거부한다.
    """

    required = {
        "split",
        "source_family",
        "native_sample_rate_hz",
        "processed_sample_rate_hz",
        "processing_role",
        "resample_count",
        "resampler_algorithm",
        "verified_resampler_passband_upper_hz",
        "resampler_frequency_response_sha256",
        "synthetic_bandwidth_claimed",
        "lossless",
        "duration_seconds",
        "crest_factor_db",
        "source_density_ratios",
        "predicted_err_density_ratios",
        "native_content_sha256",
        "processed_content_sha256",
        "transform_receipt_sha256",
        "lineage_component_id",
    }
    if set(candidate) != required:
        raise ValueError(f"source candidate key 집합이 정확하지 않습니다: {sorted(candidate)}")
    if candidate["split"] != expected_split or candidate["source_family"] != expected_family:
        raise ValueError("source candidate의 split/family가 슬롯과 다릅니다")
    rate = candidate["native_sample_rate_hz"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int)
        or rate < MIN_NATIVE_SAMPLE_RATE_HZ
    ):
        raise ValueError(
            f"source candidate native sample rate가 {MIN_NATIVE_SAMPLE_RATE_HZ}Hz 미만입니다"
        )
    processed_rate = candidate["processed_sample_rate_hz"]
    if (
        isinstance(processed_rate, bool)
        or not isinstance(processed_rate, int)
        or processed_rate != _BROADBAND_CONTRACT.sample_rate
    ):
        raise ValueError("source candidate processed playback rate가 48kHz가 아닙니다")
    processing = str(candidate["processing_role"])
    predicate = _PROCESSING_BY_NATIVE_RATE.get(processing)
    if predicate is None or not predicate(rate):
        raise ValueError("native sample rate와 processing 역할이 모순됩니다")
    count = candidate["resample_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("source candidate resample_count가 정수가 아닙니다")
    algorithm = candidate["resampler_algorithm"]
    passband = candidate["verified_resampler_passband_upper_hz"]
    response_sha = candidate["resampler_frequency_response_sha256"]
    if processing == NATIVE_EXACT_TARGET_RATE_ROLE:
        if count != 0 or algorithm is not None or passband is not None or response_sha is not None:
            raise ValueError("native exact-48k 역할에는 resampler가 없어야 합니다")
    else:
        if count != 1 or algorithm != "polyphase_fir":
            raise ValueError("native rate 변환은 polyphase FIR 정확히 1회여야 합니다")
        passband_value = float(passband)
        if not (
            math.isfinite(passband_value)
            and passband_value >= REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
            and passband_value <= min(rate / 2.0, processed_rate / 2.0)
        ):
            raise ValueError("resampler passband가 8k octave 상단을 보존하지 못합니다")
        _metadata_sha(response_sha, label="resampler frequency-response evidence")
    if candidate["synthetic_bandwidth_claimed"] is not False:
        raise ValueError("rate conversion으로 native bandwidth를 새로 주장할 수 없습니다")
    if candidate["lossless"] is not True:
        raise ValueError("source candidate는 lossless raw여야 합니다")
    duration = float(candidate["duration_seconds"])
    if not math.isfinite(duration) or duration < RECORDING_SECONDS:
        raise ValueError("source candidate window가 15초보다 짧습니다")
    crest = float(candidate["crest_factor_db"])
    if not math.isfinite(crest) or crest < 0.0 or crest > MAX_SOURCE_CREST_FACTOR_DB:
        raise ValueError("source candidate crest가 0--15dB acquisition 범위 밖입니다")
    native_digest = _metadata_sha(
        candidate["native_content_sha256"], label="source candidate native content"
    )
    processed_digest = _metadata_sha(
        candidate["processed_content_sha256"],
        label="source candidate processed content",
    )
    transform_digest = _metadata_sha(
        candidate["transform_receipt_sha256"],
        label="source candidate transform receipt",
    )
    if processing == NATIVE_EXACT_TARGET_RATE_ROLE:
        if native_digest != processed_digest:
            raise ValueError("native exact-48k source의 raw/processed SHA가 다릅니다")
    elif native_digest == processed_digest:
        raise ValueError("resampled source의 raw/processed SHA가 같습니다")
    if not str(candidate["lineage_component_id"]).strip():
        raise ValueError("source candidate lineage component가 비었습니다")

    for label in ("source_density_ratios", "predicted_err_density_ratios"):
        matrix = candidate[label]
        if not isinstance(matrix, list) or len(matrix) != CANDIDATE_SEGMENT_COUNT:
            raise ValueError(f"{label}는 9개 segment여야 합니다")
        all_band_pass = 0
        for row in matrix:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError(f"{label}의 각 segment는 7대역이어야 합니다")
            values = [float(value) for value in row]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{label}에 유효하지 않은 density가 있습니다")
            if all(value >= MIN_TARGET_D_DENSITY_RATIO for value in values):
                all_band_pass += 1
        if all_band_pass < MIN_JOINT_SEGMENTS_PER_GROUP:
            raise ValueError(f"{label}의 all-seven-band PASS segment가 8개 미만입니다")
    return {
        "status": "PASS_METADATA_ONLY",
        "split": expected_split,
        "source_family": expected_family,
        "native_content_sha256": native_digest,
        "processed_content_sha256": processed_digest,
        "transform_receipt_sha256": transform_digest,
        "lineage_component_id": str(candidate["lineage_component_id"]),
    }


def validate_campaign_candidate_set(
    plan: Mapping[str, Any],
    candidates_by_slot: Mapping[str, Mapping[str, Any]],
    *,
    forbidden_content_sha256: Sequence[str],
    forbidden_lineage_component_ids: Sequence[str],
    forbidden_processed_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """48개 후보가 실제로 서로 독립이고 기존 corpus와 분리됐는지 재검사한다."""

    if plan.get("schema") != BROADBAND_RECORDING_CAMPAIGN_SCHEMA:
        raise ValueError("broadband recording campaign schema가 다릅니다")
    slots = plan.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("campaign slot이 비었습니다")
    slot_ids = [str(slot.get("slot_id", "")) for slot in slots if isinstance(slot, Mapping)]
    if len(slot_ids) != len(slots) or any(not slot for slot in slot_ids):
        raise ValueError("campaign slot id가 비었거나 형식이 잘못됐습니다")
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("campaign slot id가 중복입니다")
    if set(candidates_by_slot) != set(slot_ids):
        raise ValueError("candidate set이 campaign slot과 전단사 관계가 아닙니다")

    forbidden_content = {str(value) for value in forbidden_content_sha256}
    forbidden_processed = {str(value) for value in forbidden_processed_sha256}
    forbidden_lineage = {str(value) for value in forbidden_lineage_component_ids}
    seen_native_content: set[str] = set()
    seen_processed_content: set[str] = set()
    seen_transform_receipts: set[str] = set()
    seen_lineage: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for slot in slots:
        assert isinstance(slot, Mapping)
        slot_id = str(slot["slot_id"])
        split = str(slot.get("split", ""))
        family = str(slot.get("source_family", ""))
        result = validate_source_candidate_metadata(
            candidates_by_slot[slot_id],
            expected_split=split,
            expected_family=family,
        )
        native_content = str(result["native_content_sha256"])
        processed_content = str(result["processed_content_sha256"])
        transform_receipt = str(result["transform_receipt_sha256"])
        lineage = str(result["lineage_component_id"])
        if native_content in forbidden_content:
            raise ValueError(f"기존 recorded/synthetic content를 재사용했습니다: {slot_id}")
        if processed_content in forbidden_processed:
            raise ValueError(f"기존 processed content를 재사용했습니다: {slot_id}")
        if lineage in forbidden_lineage:
            raise ValueError(f"기존 recorded/synthetic lineage를 재사용했습니다: {slot_id}")
        if native_content in seen_native_content:
            raise ValueError("같은 native source content를 여러 신규 group으로 위장했습니다")
        if processed_content in seen_processed_content:
            raise ValueError("같은 processed source content를 여러 신규 group으로 위장했습니다")
        if transform_receipt in seen_transform_receipts:
            raise ValueError("같은 source transform receipt를 여러 신규 group에 재사용했습니다")
        if lineage in seen_lineage:
            raise ValueError("같은 artist/speaker/machine lineage를 여러 group으로 위장했습니다")
        seen_native_content.add(native_content)
        seen_processed_content.add(processed_content)
        seen_transform_receipts.add(transform_receipt)
        seen_lineage.add(lineage)
        counts[(split, family)] += 1
    expected = {
        (str(row["split"]), str(row["source_family"])): int(
            row["minimum_new_all_band_groups"]
        )
        for row in plan["minimum"]["rows"]
    }
    if dict(counts) != expected:
        raise ValueError("candidate set의 split×family 수가 최소 캠페인과 다릅니다")
    return {
        "status": "PASS_METADATA_SET_ONLY",
        "candidate_count": len(candidates_by_slot),
        "unique_native_content_count": len(seen_native_content),
        "unique_processed_content_count": len(seen_processed_content),
        "unique_transform_receipt_count": len(seen_transform_receipts),
        "unique_lineage_component_count": len(seen_lineage),
    }


__all__ = [
    "BROADBAND_RECORDING_CAMPAIGN_SCHEMA",
    "CANDIDATE_SEGMENT_COUNT",
    "MAX_SOURCE_CREST_FACTOR_DB",
    "MIN_NATIVE_SAMPLE_RATE_HZ",
    "RECORDING_SECONDS",
    "REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ",
    "SEGMENT_SECONDS",
    "SEGMENT_START_OFFSET_SECONDS",
    "build_missing_source_campaign",
    "calculate_minimum_new_groups",
    "source_candidate_requirements",
    "validate_campaign_candidate_set",
    "validate_source_candidate_metadata",
]
