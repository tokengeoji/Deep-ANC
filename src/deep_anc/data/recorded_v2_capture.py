"""광대역 recorded-v2 실측 수집의 fail-closed 계약.

이 모듈은 오디오 장치를 import하거나 열지 않는다. 역사적 ``record_duct`` 세션을
광대역 세대로 이름만 바꾸는 경로를 막고, 향후 live recorder가 남겨야 할 source/plant/
raw/time-warp/session 증거를 검증한다.

핵심 경계는 다음과 같다.

* source 한 행은 실제 제출되는 48 kHz, 15초 int16 PCM bytes까지 사전에 봉인한다.
* canonical fullband causal P/S의 **외부 file SHA anchor**가 없으면 plan도 BLOCKED다.
* 시간축은 actual submitted PCM과 ERR/REF가 공유하는 absolute DAC-q map이어야 한다.
  600 Hz 위 신호는 clock fit/phase repair에 사용할 수 없다.
* live raw는 analysis보다 먼저 별도 no-replace directory에 발행한다.
* ``source_aligned.wav``/``mics.wav``의 coverage는 저장 scalar를 믿지 않고 실제 ERR
  ch0에서 일곱 대역을 다시 계산한다.

구조 검증 PASS는 live authority가 아니다. ``RECORDED_V2_LIVE_AUTHORITY``가 ``None``인
동안 어떤 fixture나 self-sealed JSON도 스피커 출력을 열 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..dsp.control_band_contract import (
    ControlBandContract,
    max_timing_error_samples_for_attenuation,
)
from .atomic_publish import publish_directory_noreplace
from .broadband_coverage_receipt import (
    MIN_SOURCE_ERR_COHERENCE,
    MIN_TARGET_D_DENSITY_RATIO,
    _validate_source_transform,
)
from .recorded_broadband_coverage import measure_broadband_session


RECORDED_V2_CAPTURE_CONTRACT_SCHEMA = "recorded_broadband_v2_capture_contract_v1"
RECORDED_V2_SOURCE_PLAN_SCHEMA = "recorded_broadband_v2_source_plan_v1"
RECORDED_V2_CAMPAIGN_SELECTION_SCHEMA = (
    "recorded_broadband_v2_campaign_selection_receipt_v1"
)
RECORDED_V2_CANDIDATE_SET_SCHEMA = "recorded_broadband_v2_candidate_set_receipt_v1"
RECORDED_V2_RAW_CAPTURE_SCHEMA = "recorded_broadband_v2_raw_capture_v1"
RECORDED_V2_TIMEWARP_SCHEMA = "recorded_broadband_v2_absolute_dac_q_timewarp_v1"
RECORDED_V2_SESSION_SCHEMA = "recorded_broadband_v2_session_v1"
FULLBAND_CAUSAL_PLANT_SCHEMA = "fullband_causal_physical_plant_evidence_v1"

# root가 exact plan/plant/source bytes와 전체 회귀를 검토하기 전에는 바꾸지 않는다.
RECORDED_V2_LIVE_AUTHORITY: dict[str, str] | None = None

SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
SOURCE_SECONDS = 15.0
SOURCE_FRAMES = int(SAMPLE_RATE * SOURCE_SECONDS)
FADE_FRAMES = int(0.1 * SAMPLE_RATE)
QUANTIZER = "clip_-1_1_mul_32767_rint_ties_to_even_little_endian_int16_v1"
MAX_SUBMITTED_PEAK_INT16 = 4_915  # round(0.15 * 32767), 공용 recording peak 상한
CLOCK_FIT_BAND_HZ = (152.0, 600.0)
CLOCK_MIN_FIT_WINDOWS = 8
CLOCK_MIN_HOLDOUT_WINDOWS = 8
CLOCK_LEAVEOUT_MAX_SAMPLES = 0.050
CLOCK_CUBIC_MAX_SAMPLES = 0.006
CLOCK_COMBINED_MAX_SAMPLES = 0.056
REQUIRED_WITNESS_ROLES = ("P_submitted_playback", "ERR_ch0", "REF_ch1")
REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_FAMILIES = ("speech", "music", "environment", "machine")
MIN_GROUPS_PER_CELL = 4
_HEX = frozenset("0123456789abcdef")


class RecordedV2Blocked(RuntimeError):
    """필요한 외부 실측/소스 증거가 아직 없어 다음 단계를 열 수 없음."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위가 object가 아닙니다: {path}")
    return value


def _regular_file_reference(
    value: object,
    *,
    repository_root: Path,
    label: str,
    require_local: bool = True,
) -> dict[str, Any]:
    row = _exact_keys(value, {"path", "size_bytes", "sha256"}, label=label)
    text = str(row["path"])
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path는 정규화된 저장소 상대경로여야 합니다")
    candidate = Path(os.path.abspath(repository_root / relative))
    try:
        candidate.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label}.path가 저장소 밖입니다") from exc
    current = repository_root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label}.path에 symlink가 있습니다")
    size = row["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_bytes가 양의 정수가 아닙니다")
    digest = _sha(row["sha256"], label=f"{label}.sha256")
    if require_local:
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"{label} regular file이 없습니다: {candidate}")
        if candidate.stat().st_size != size or sha256_file(candidate) != digest:
            raise ValueError(f"{label} bytes가 receipt와 다릅니다")
    return {"path": text, "size_bytes": int(size), "sha256": digest}


def file_reference(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    candidate = Path(path).resolve(strict=True)
    relative = candidate.relative_to(root)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("file reference는 symlink가 아닌 regular file이어야 합니다")
    return {
        "path": relative.as_posix(),
        "size_bytes": candidate.stat().st_size,
        "sha256": sha256_file(candidate),
    }


def capture_contract() -> dict[str, Any]:
    control = ControlBandContract.broadband_point_control()
    payload: dict[str, Any] = {
        "schema": RECORDED_V2_CAPTURE_CONTRACT_SCHEMA,
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "source_seconds": SOURCE_SECONDS,
        "source_frames": SOURCE_FRAMES,
        "submitted_pcm_dtype": "little_endian_int16",
        "submitted_pcm_channels": {"noise_out": 0, "cancel_out_silent": 1},
        "quantizer": QUANTIZER,
        "maximum_submitted_peak_int16": MAX_SUBMITTED_PEAK_INT16,
        "fade_frames_each_edge": FADE_FRAMES,
        "schemas": {
            "source_plan": RECORDED_V2_SOURCE_PLAN_SCHEMA,
            "campaign_selection": RECORDED_V2_CAMPAIGN_SELECTION_SCHEMA,
            "candidate_set": RECORDED_V2_CANDIDATE_SET_SCHEMA,
            "physical_fullband_plant": FULLBAND_CAUSAL_PLANT_SCHEMA,
            "raw_capture": RECORDED_V2_RAW_CAPTURE_SCHEMA,
            "timewarp": RECORDED_V2_TIMEWARP_SCHEMA,
            "session": RECORDED_V2_SESSION_SCHEMA,
        },
        "clock": {
            "fit_band_hz": list(CLOCK_FIT_BAND_HZ),
            "highband_used_for_clock_fit": False,
            "highband_phase_repair_samples": 0.0,
            "minimum_fit_windows": CLOCK_MIN_FIT_WINDOWS,
            "minimum_holdout_windows": CLOCK_MIN_HOLDOUT_WINDOWS,
            "leaveout_max_samples": CLOCK_LEAVEOUT_MAX_SAMPLES,
            "cubic_max_samples": CLOCK_CUBIC_MAX_SAMPLES,
            "combined_max_samples": CLOCK_COMBINED_MAX_SAMPLES,
            "required_common_witness_roles": list(REQUIRED_WITNESS_ROLES),
            "callback_role": "monotonic_and_slip_witness_only",
        },
        "control_band_contract_sha256": control.digest(),
        "point_control_subbands_hz": [
            list(band) for band in control.point_control_subbands_hz
        ],
        "raw_first_no_replace_required": True,
        "analysis_after_raw_publish_only": True,
        "device_occupancy_evidence_file_required_at_two_live_boundaries": True,
    }
    payload["contract_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _sealed_payload(value: object, *, expected_schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise ValueError(f"{label} schema가 다릅니다")
    evidence = _sha(value.get("evidence_sha256"), label=f"{label} evidence_sha256")
    expected = _sha256_bytes(
        _canonical_json({key: item for key, item in value.items() if key != "evidence_sha256"})
    )
    if evidence != expected:
        raise ValueError(f"{label} evidence_sha256이 payload 재계산과 다릅니다")
    return dict(value)


def _validate_device_occupancy_witness(
    value: object, *, repository_root: Path
) -> dict[str, Any]:
    row = _exact_keys(
        value,
        {
            "checked_before_input_probe",
            "checked_before_output_open",
            "all_pcm_closed",
            "evidence_file",
        },
        label="raw device occupancy witness",
    )
    if (
        row["checked_before_input_probe"] is not True
        or row["checked_before_output_open"] is not True
        or row["all_pcm_closed"] is not True
    ):
        raise ValueError("raw capture 직전 장치 무점유 증거가 없습니다")
    reference = _regular_file_reference(
        row["evidence_file"],
        repository_root=repository_root,
        label="raw device occupancy evidence file",
    )
    payload = _sealed_payload(
        _json_no_duplicates(repository_root / reference["path"]),
        expected_schema="recorded_v2_audio_occupancy_witness_v1",
        label="raw device occupancy evidence",
    )
    _exact_keys(
        payload,
        {"schema", "checks", "all_pcm_closed", "evidence_sha256"},
        label="raw device occupancy evidence",
    )
    checks = payload["checks"]
    if not isinstance(checks, list) or len(checks) != 2:
        raise ValueError("device occupancy evidence는 두 live 경계여야 합니다")
    expected_stages = {"before_input_probe", "before_output_open"}
    observed_stages: set[str] = set()
    for raw in checks:
        check = _exact_keys(
            raw,
            {"stage", "proc_pcm_status", "fuser_pcm_owners"},
            label="device occupancy check",
        )
        stage = str(check["stage"])
        observed_stages.add(stage)
        statuses = check["proc_pcm_status"]
        if not isinstance(statuses, list) or not statuses:
            raise ValueError("device occupancy /proc PCM status가 비었습니다")
        for status in statuses:
            status_row = _exact_keys(
                status, {"path", "status"}, label="device occupancy PCM status"
            )
            if not str(status_row["path"]).startswith("/proc/asound/") or str(
                status_row["status"]
            ).strip().lower() != "closed":
                raise ValueError("device occupancy evidence에 열린 PCM이 있습니다")
        if check["fuser_pcm_owners"] != []:
            raise ValueError("device occupancy evidence에 PCM owner가 있습니다")
    if observed_stages != expected_stages or payload["all_pcm_closed"] is not True:
        raise ValueError("device occupancy stage/all-closed 증거가 다릅니다")
    return reference


def validate_fullband_causal_plant(
    plant_path: str | Path,
    *,
    expected_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """외부 SHA로 고정한 physical fullband causal P/S만 recorded-v2에 허용한다."""

    root = Path(repository_root).resolve(strict=True)
    path = Path(plant_path)
    path = path if path.is_absolute() else root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise RecordedV2Blocked("canonical fullband causal plant evidence가 없습니다") from exc
    expected_sha = _sha(expected_file_sha256, label="expected plant file SHA")
    actual_sha = sha256_file(resolved)
    if actual_sha != expected_sha:
        raise RecordedV2Blocked(
            f"fullband plant 외부 SHA 불일치: expected={expected_sha}, actual={actual_sha}"
        )
    payload = _sealed_payload(
        _json_no_duplicates(resolved),
        expected_schema=FULLBAND_CAUSAL_PLANT_SCHEMA,
        label="fullband causal plant",
    )
    required = {
        "schema",
        "role",
        "status",
        "canonical_training_eligible",
        "synthetic_fixture",
        "control_band_contract_sha256",
        "sample_rate_hz",
        "block_size",
        "excitation_band_hz",
        "persistently_exciting_causal_history",
        "same_capture_ps",
        "raw_capture",
        "analysis",
        "primary_path",
        "secondary_path",
        "absolute_dac_q_timewarp",
        "training_timing_contract_sha256",
        "evidence_sha256",
    }
    _exact_keys(payload, required, label="fullband causal plant")
    contract = ControlBandContract.broadband_point_control()
    if (
        payload["role"] != "physical_live_raw_first_production"
        or payload["status"] != "PASS"
        or payload["canonical_training_eligible"] is not True
        or payload["synthetic_fixture"] is not False
        or payload["persistently_exciting_causal_history"] is not True
        or payload["same_capture_ps"] is not True
    ):
        raise RecordedV2Blocked("physical causal P/S canonical 자격이 PASS가 아닙니다")
    if payload["control_band_contract_sha256"] != contract.digest():
        raise RecordedV2Blocked("plant control-band contract SHA가 현재 광대역 계약과 다릅니다")
    if payload["sample_rate_hz"] != SAMPLE_RATE or payload["block_size"] != BLOCK_SIZE:
        raise RecordedV2Blocked("plant sample-rate/block 계약이 48k/256이 아닙니다")
    band = payload["excitation_band_hz"]
    if (
        not isinstance(band, list)
        or len(band) != 2
        or float(band[0]) > contract.required_excitation_lower_hz
        or float(band[1]) < contract.required_excitation_upper_hz
    ):
        raise RecordedV2Blocked("plant excitation이 150--11.314kHz를 덮지 못합니다")
    refs = {}
    for name in ("raw_capture", "analysis", "primary_path", "secondary_path", "absolute_dac_q_timewarp"):
        refs[name] = _regular_file_reference(
            payload[name], repository_root=root, label=f"plant.{name}"
        )
    timing_sha = _sha(
        payload["training_timing_contract_sha256"], label="plant training timing SHA"
    )
    return {
        "status": "PASS_STRUCTURAL_OFFLINE_ONLY",
        "file": file_reference(resolved, repository_root=root),
        "payload_evidence_sha256": payload["evidence_sha256"],
        "training_timing_contract_sha256": timing_sha,
        "references": refs,
    }


def render_submitted_pcm(
    processed_path: str | Path,
    *,
    start_frame: int,
    gain_q15: int,
) -> tuple[np.ndarray, np.ndarray]:
    """source window를 live callback과 동일한 int16 mono/stereo bytes로 만든다."""

    import soundfile as sf

    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame은 0 이상 정수여야 합니다")
    if isinstance(gain_q15, bool) or not isinstance(gain_q15, int) or not 1 <= gain_q15 <= 32767:
        raise ValueError("gain_q15는 1--32767 정수여야 합니다")
    path = Path(processed_path)
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError("processed playback source는 실제 48kHz mono여야 합니다")
    if start_frame + SOURCE_FRAMES > info.frames:
        raise ValueError("15초 playback window가 processed source 길이를 넘습니다")
    with sf.SoundFile(str(path), "r") as handle:
        handle.seek(start_frame)
        values = handle.read(SOURCE_FRAMES, dtype="float64", always_2d=False)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (SOURCE_FRAMES,) or not np.all(np.isfinite(values)):
        raise ValueError("processed source decode가 exact 15초 finite mono가 아닙니다")
    envelope = np.ones(SOURCE_FRAMES, dtype=np.float64)
    ramp = np.arange(FADE_FRAMES, dtype=np.float64) / float(FADE_FRAMES)
    envelope[:FADE_FRAMES] = ramp
    envelope[-FADE_FRAMES:] = ramp[::-1]
    scaled = values * (gain_q15 / 32767.0) * envelope
    mono = np.rint(np.clip(scaled, -1.0, 1.0) * 32767.0).astype("<i2")
    stereo = np.zeros((SOURCE_FRAMES, 2), dtype="<i2")
    stereo[:, 0] = mono
    return mono, stereo


def submitted_pcm_evidence(mono: np.ndarray, stereo: np.ndarray) -> dict[str, Any]:
    one_raw = np.asarray(mono)
    two_raw = np.asarray(stereo)
    if one_raw.dtype != np.dtype("<i2") or two_raw.dtype != np.dtype("<i2"):
        raise ValueError("submitted PCM은 변환 전부터 little-endian int16이어야 합니다")
    one = np.ascontiguousarray(one_raw)
    two = np.ascontiguousarray(two_raw)
    if one.shape != (SOURCE_FRAMES,) or two.shape != (SOURCE_FRAMES, 2):
        raise ValueError("submitted PCM shape가 exact 15초 mono/stereo가 아닙니다")
    if not np.array_equal(two[:, 0], one) or np.any(two[:, 1] != 0):
        raise ValueError("submitted stereo ch0/source 또는 silent ch1 계약이 다릅니다")
    peak = int(np.max(np.abs(one.astype(np.int32))))
    if peak > MAX_SUBMITTED_PEAK_INT16:
        raise ValueError(
            f"submitted source peak {peak}가 안전 상한 {MAX_SUBMITTED_PEAK_INT16}을 넘습니다"
        )
    return {
        "source_frames": SOURCE_FRAMES,
        "mono_pcm_sha256": _sha256_bytes(one.tobytes(order="C")),
        "stereo_interleaved_pcm_sha256": _sha256_bytes(two.tobytes(order="C")),
        "peak_int16": peak,
        "cancel_channel_all_zero": True,
        "dtype": "little_endian_int16",
        "quantizer": QUANTIZER,
    }


def validate_source_plan(
    plan_path: str | Path,
    *,
    expected_plan_sha256: str,
    expected_plant_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """실제 source bytes를 다시 decode해 plan의 submitted PCM SHA까지 검증한다."""

    root = Path(repository_root).resolve(strict=True)
    path = Path(plan_path)
    path = path if path.is_absolute() else root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise RecordedV2Blocked("recorded-v2 verified source plan이 없습니다") from exc
    expected_plan = _sha(expected_plan_sha256, label="expected source-plan SHA")
    if sha256_file(resolved) != expected_plan:
        raise RecordedV2Blocked("recorded-v2 source plan 외부 SHA가 다릅니다")
    plan = _sealed_payload(
        _json_no_duplicates(resolved),
        expected_schema=RECORDED_V2_SOURCE_PLAN_SCHEMA,
        label="recorded-v2 source plan",
    )
    _exact_keys(
        plan,
        {
            "schema",
            "role",
            "status",
            "capture_contract_sha256",
            "control_band_contract_sha256",
            "plant_evidence",
            "campaign_receipt",
            "candidate_set_receipt",
            "sessions",
            "evidence_sha256",
        },
        label="recorded-v2 source plan",
    )
    if plan["role"] != "physical_recorded_v2_live_source_plan" or plan["status"] != "READY":
        raise RecordedV2Blocked("source plan이 physical READY 역할이 아닙니다")
    contract = capture_contract()
    control = ControlBandContract.broadband_point_control()
    if (
        plan["capture_contract_sha256"] != contract["contract_sha256"]
        or plan["control_band_contract_sha256"] != control.digest()
    ):
        raise RecordedV2Blocked("source plan의 capture/control contract SHA가 다릅니다")
    plant_ref = _regular_file_reference(
        plan["plant_evidence"], repository_root=root, label="source plan plant"
    )
    if plant_ref["sha256"] != _sha(expected_plant_sha256, label="expected plant SHA"):
        raise RecordedV2Blocked("source plan과 외부 plant SHA anchor가 다릅니다")
    plant_result = validate_fullband_causal_plant(
        plant_ref["path"],
        expected_file_sha256=expected_plant_sha256,
        repository_root=root,
    )
    campaign_ref = _regular_file_reference(
        plan["campaign_receipt"], repository_root=root, label="source plan campaign receipt"
    )
    candidate_ref = _regular_file_reference(
        plan["candidate_set_receipt"],
        repository_root=root,
        label="source plan candidate-set receipt",
    )
    sessions = plan["sessions"]
    if not isinstance(sessions, list) or len(sessions) < 48:
        raise RecordedV2Blocked("recorded-v2 source plan은 최소 48개 verified source가 필요합니다")
    seen_session: set[str] = set()
    seen_group: set[str] = set()
    seen_lineage: set[str] = set()
    seen_native_sha: set[str] = set()
    seen_processed_sha: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(sessions):
        row = _exact_keys(
            raw,
            {
                "session_id",
                "split",
                "source_family",
                "group_id",
                "lineage_id",
                "source",
                "window",
                "playback",
            },
            label=f"source plan session#{index}",
        )
        session_id = str(row["session_id"]).strip()
        split = str(row["split"])
        family = str(row["source_family"])
        group = str(row["group_id"]).strip()
        lineage = str(row["lineage_id"]).strip()
        if not session_id or session_id in seen_session:
            raise ValueError("source plan session_id가 비었거나 중복입니다")
        if split not in REQUIRED_SPLITS or family not in REQUIRED_FAMILIES:
            raise ValueError(f"{session_id}: split/family가 canonical 집합 밖입니다")
        if not group or group in seen_group or not lineage or lineage in seen_lineage:
            raise ValueError(f"{session_id}: group/lineage가 비었거나 독립적이지 않습니다")
        seen_session.add(session_id)
        seen_group.add(group)
        seen_lineage.add(lineage)
        raw_native, processed, transform = _validate_source_transform(
            row["source"],
            contract=control,
            root=root,
            session_id=session_id,
            require_local_files=True,
        )
        if raw_native["sha256"] in seen_native_sha or processed["sha256"] in seen_processed_sha:
            raise ValueError(f"{session_id}: native/processed source content가 중복입니다")
        seen_native_sha.add(raw_native["sha256"])
        seen_processed_sha.add(processed["sha256"])
        window = _exact_keys(row["window"], {"start_frame", "n_frames"}, label=f"{session_id}.window")
        start = window["start_frame"]
        n_frames = window["n_frames"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or n_frames != SOURCE_FRAMES
        ):
            raise ValueError(f"{session_id}: source window가 exact 15초가 아닙니다")
        playback = _exact_keys(
            row["playback"],
            {
                "gain_q15",
                "fade_frames_each_edge",
                "quantizer",
                "submitted_pcm_evidence",
            },
            label=f"{session_id}.playback",
        )
        if playback["fade_frames_each_edge"] != FADE_FRAMES or playback["quantizer"] != QUANTIZER:
            raise ValueError(f"{session_id}: fade/quantizer 계약이 다릅니다")
        mono, stereo = render_submitted_pcm(
            root / processed["path"], start_frame=start, gain_q15=playback["gain_q15"]
        )
        evidence = submitted_pcm_evidence(mono, stereo)
        if playback["submitted_pcm_evidence"] != evidence:
            raise ValueError(f"{session_id}: actual submitted int16 PCM SHA/peak가 plan과 다릅니다")
        counts[(split, family)] += 1
        rows.append(
            {
                "session_id": session_id,
                "split": split,
                "source_family": family,
                "group_id": group,
                "lineage_id": lineage,
                "native_source_sha256": raw_native["sha256"],
                "processed_source_sha256": processed["sha256"],
                "transform_receipt_sha256": transform["sha256"],
                "submitted_mono_pcm_sha256": evidence["mono_pcm_sha256"],
                "submitted_stereo_pcm_sha256": evidence["stereo_interleaved_pcm_sha256"],
                "source_plan_row_sha256": _sha256_bytes(_canonical_json(row)),
            }
        )
    missing = [
        f"{split}/{family}={counts[(split, family)]}"
        for split in REQUIRED_SPLITS
        for family in REQUIRED_FAMILIES
        if counts[(split, family)] < MIN_GROUPS_PER_CELL
    ]
    if missing:
        raise RecordedV2Blocked("source plan split×family 독립 group이 4 미만입니다: " + ", ".join(missing))
    campaign = _sealed_payload(
        _json_no_duplicates(root / campaign_ref["path"]),
        expected_schema=RECORDED_V2_CAMPAIGN_SELECTION_SCHEMA,
        label="recorded-v2 campaign selection",
    )
    _exact_keys(
        campaign,
        {
            "schema",
            "role",
            "status",
            "control_band_contract_sha256",
            "plant_evidence_sha256",
            "minimum_session_count",
            "evidence_sha256",
        },
        label="recorded-v2 campaign selection",
    )
    minimum_session_count = campaign["minimum_session_count"]
    if (
        campaign["role"] != "physical_candidate_selection_after_causal_plant"
        or campaign["status"] != "PASS"
        or campaign["control_band_contract_sha256"] != control.digest()
        or campaign["plant_evidence_sha256"] != plant_ref["sha256"]
        or isinstance(minimum_session_count, bool)
        or not isinstance(minimum_session_count, int)
        or minimum_session_count < 48
    ):
        raise RecordedV2Blocked("campaign selection receipt가 physical fullband PASS가 아닙니다")
    candidate = _sealed_payload(
        _json_no_duplicates(root / candidate_ref["path"]),
        expected_schema=RECORDED_V2_CANDIDATE_SET_SCHEMA,
        label="recorded-v2 candidate set",
    )
    _exact_keys(
        candidate,
        {
            "schema",
            "role",
            "status",
            "control_band_contract_sha256",
            "campaign_receipt_sha256",
            "recorded_lineage_authority",
            "synthetic_lineage_authority",
            "sessions",
            "evidence_sha256",
        },
        label="recorded-v2 candidate set",
    )
    if (
        candidate["role"] != "lineage_dsu_and_native_transform_verified"
        or candidate["status"] != "PASS"
        or candidate["control_band_contract_sha256"] != control.digest()
        or candidate["campaign_receipt_sha256"] != campaign_ref["sha256"]
    ):
        raise RecordedV2Blocked("candidate set receipt가 lineage/native transform PASS가 아닙니다")
    recorded_lineage_ref = _regular_file_reference(
        candidate["recorded_lineage_authority"],
        repository_root=root,
        label="candidate recorded lineage authority",
    )
    synthetic_lineage_ref = _regular_file_reference(
        candidate["synthetic_lineage_authority"],
        repository_root=root,
        label="candidate synthetic lineage authority",
    )
    candidate_sessions = candidate["sessions"]
    if not isinstance(candidate_sessions, list):
        raise ValueError("candidate set sessions가 list가 아닙니다")
    expected_candidate_rows = [
        {
            key: row[key]
            for key in (
                "session_id",
                "split",
                "source_family",
                "group_id",
                "lineage_id",
                "native_source_sha256",
                "processed_source_sha256",
                "transform_receipt_sha256",
            )
        }
        for row in rows
    ]
    for index, row in enumerate(candidate_sessions):
        _exact_keys(
            row,
            set(expected_candidate_rows[0]),
            label=f"candidate set session#{index}",
        )
    sort_key = lambda row: str(row["session_id"])
    if sorted(candidate_sessions, key=sort_key) != sorted(
        expected_candidate_rows, key=sort_key
    ):
        raise ValueError("candidate set과 source plan의 source/split/family/lineage exact 집합이 다릅니다")
    return {
        "status": "PASS_STRUCTURAL_OFFLINE_ONLY",
        "live_authority": RECORDED_V2_LIVE_AUTHORITY,
        "plan": file_reference(resolved, repository_root=root),
        "plant": plant_result,
        "campaign_receipt_sha256": campaign_ref["sha256"],
        "candidate_set_receipt_sha256": candidate_ref["sha256"],
        "recorded_lineage_authority_sha256": recorded_lineage_ref["sha256"],
        "synthetic_lineage_authority_sha256": synthetic_lineage_ref["sha256"],
        "session_count": len(rows),
        "cell_counts": {
            f"{split}/{family}": counts[(split, family)]
            for split in REQUIRED_SPLITS
            for family in REQUIRED_FAMILIES
        },
        "sessions": rows,
    }


def validate_timewarp_receipt(
    receipt: object,
    *,
    repository_root: str | Path,
    expected_raw_capture_sha256: str,
    expected_submitted_pcm_sha256: str,
    expected_mics_pcm_sha256: str,
) -> dict[str, Any]:
    """P/ERR/REF가 공유하는 low-band-only absolute DAC-q map을 검증한다."""

    root = Path(repository_root).resolve(strict=True)
    value = _sealed_payload(
        receipt, expected_schema=RECORDED_V2_TIMEWARP_SCHEMA, label="recorded-v2 timewarp"
    )
    _exact_keys(
        value,
        {
            "schema",
            "role",
            "raw_capture_sha256",
            "submitted_pcm_sha256",
            "mics_pcm_sha256",
            "fit_band_hz",
            "highband_used_for_clock_fit",
            "highband_phase_repair_samples",
            "fit_window_parity",
            "holdout_window_parity",
            "holdout_used_for_fit_or_selection",
            "common_map",
            "witnesses",
            "callback_witness",
            "sample_slip_count",
            "xrun_count",
            "evidence_sha256",
        },
        label="recorded-v2 timewarp",
    )
    if value["role"] != "physical_raw_offline_alignment":
        raise ValueError("timewarp receipt가 physical raw 역할이 아닙니다")
    anchors = (
        ("raw capture", value["raw_capture_sha256"], expected_raw_capture_sha256),
        ("submitted PCM", value["submitted_pcm_sha256"], expected_submitted_pcm_sha256),
        ("mics PCM", value["mics_pcm_sha256"], expected_mics_pcm_sha256),
    )
    for label, actual, expected in anchors:
        if _sha(actual, label=f"timewarp {label} SHA") != _sha(expected, label=f"expected {label} SHA"):
            raise ValueError(f"timewarp {label} lineage가 다릅니다")
    if list(value["fit_band_hz"]) != list(CLOCK_FIT_BAND_HZ):
        raise ValueError("clock fit band는 exact 152--600Hz여야 합니다")
    if value["highband_used_for_clock_fit"] is not False or float(
        value["highband_phase_repair_samples"]
    ) != 0.0:
        raise ValueError("고역을 clock fit/phase repair에 사용할 수 없습니다")
    if (
        value["fit_window_parity"] != "even"
        or value["holdout_window_parity"] != "odd"
        or value["holdout_used_for_fit_or_selection"] is not False
    ):
        raise ValueError("clock fit/holdout 분리가 exact even/odd가 아닙니다")
    common = _exact_keys(
        value["common_map"],
        {
            "method",
            "knots_file",
            "adc_frame_knots_sha256",
            "dac_q_knots_sha256",
            "map_sha256",
            "leaveout_max_samples",
            "cubic_crosscheck_max_samples",
            "combined_max_samples",
        },
        label="timewarp common_map",
    )
    if common["method"] != "monotone_piecewise_cubic_absolute_adc_to_dac_q_v1":
        raise ValueError("absolute DAC-q map interpolation 방법이 다릅니다")
    knots_ref = _regular_file_reference(
        common["knots_file"], repository_root=root, label="timewarp knots"
    )
    with np.load(root / knots_ref["path"], allow_pickle=False) as archive:
        if set(archive.files) != {"adc_frame_knots", "dac_q_knots"}:
            raise ValueError("timewarp knots NPZ key 집합이 정확하지 않습니다")
        adc = np.asarray(archive["adc_frame_knots"], dtype="<f8")
        dac = np.asarray(archive["dac_q_knots"], dtype="<f8")
    if (
        adc.ndim != 1
        or dac.shape != adc.shape
        or adc.size < CLOCK_MIN_FIT_WINDOWS + CLOCK_MIN_HOLDOUT_WINDOWS
        or not np.all(np.isfinite(adc))
        or not np.all(np.isfinite(dac))
        or not np.all(np.diff(adc) > 0.0)
        or not np.all(np.diff(dac) > 0.0)
    ):
        raise ValueError("timewarp knots가 finite monotone common map이 아닙니다")
    if _sha256_bytes(adc.tobytes(order="C")) != _sha(
        common["adc_frame_knots_sha256"], label="ADC knot SHA"
    ) or _sha256_bytes(dac.tobytes(order="C")) != _sha(
        common["dac_q_knots_sha256"], label="DAC-q knot SHA"
    ):
        raise ValueError("timewarp knot array SHA가 다릅니다")
    map_payload = {
        "method": common["method"],
        "knots_file_sha256": knots_ref["sha256"],
        "adc_frame_knots_sha256": common["adc_frame_knots_sha256"],
        "dac_q_knots_sha256": common["dac_q_knots_sha256"],
    }
    if _sha256_bytes(_canonical_json(map_payload)) != _sha(
        common["map_sha256"], label="common map SHA"
    ):
        raise ValueError("timewarp common map SHA가 재계산과 다릅니다")
    limits = (
        ("leaveout", common["leaveout_max_samples"], CLOCK_LEAVEOUT_MAX_SAMPLES),
        ("cubic", common["cubic_crosscheck_max_samples"], CLOCK_CUBIC_MAX_SAMPLES),
        ("combined", common["combined_max_samples"], CLOCK_COMBINED_MAX_SAMPLES),
    )
    for label, actual, maximum in limits:
        number = float(actual)
        if not math.isfinite(number) or number < 0.0 or number > maximum:
            raise ValueError(f"timewarp {label} error가 hard budget을 넘습니다")
    witnesses = value["witnesses"]
    if (
        not isinstance(witnesses, list)
        or len(witnesses) != len(REQUIRED_WITNESS_ROLES)
        or {row.get("role") for row in witnesses if isinstance(row, dict)}
        != set(REQUIRED_WITNESS_ROLES)
    ):
        raise ValueError("P/ERR/REF common clock witness exact 집합이 아닙니다")
    map_sha = common["map_sha256"]
    for raw in witnesses:
        row = _exact_keys(
            raw,
            {
                "role",
                "common_map_sha256",
                "fit_windows",
                "holdout_windows",
                "maximum_residual_samples",
                "minimum_score",
            },
            label="timewarp witness",
        )
        if row["common_map_sha256"] != map_sha:
            raise ValueError("P/ERR/REF가 같은 absolute DAC-q map을 공유하지 않습니다")
        fit_windows = row["fit_windows"]
        holdout_windows = row["holdout_windows"]
        if (
            isinstance(fit_windows, bool)
            or not isinstance(fit_windows, int)
            or isinstance(holdout_windows, bool)
            or not isinstance(holdout_windows, int)
            or fit_windows < CLOCK_MIN_FIT_WINDOWS
            or holdout_windows < CLOCK_MIN_HOLDOUT_WINDOWS
        ):
            raise ValueError("clock fit/holdout window가 8개 미만입니다")
        residual = float(row["maximum_residual_samples"])
        score = float(row["minimum_score"])
        if not (math.isfinite(residual) and 0.0 <= residual <= CLOCK_COMBINED_MAX_SAMPLES):
            raise ValueError("clock witness residual이 budget을 넘습니다")
        if not (math.isfinite(score) and 0.995 <= score <= 1.0):
            raise ValueError("clock witness score가 0.995 미만입니다")
    callback = _exact_keys(
        value["callback_witness"],
        {"role", "monotonic", "sample_slip_count"},
        label="timewarp callback witness",
    )
    if (
        callback["role"] != "monotonic_and_slip_witness_only"
        or callback["monotonic"] is not True
        or callback["sample_slip_count"] != 0
        or value["sample_slip_count"] != 0
        or value["xrun_count"] != 0
    ):
        raise ValueError("callback/xrun/sample-slip witness가 PASS가 아닙니다")
    control = ControlBandContract.broadband_point_control()
    per_band_budget = [
        max_timing_error_samples_for_attenuation(
            control.measurement_resolution_attenuation_db,
            band[1],
            control.sample_rate,
        )
        for band in control.point_control_subbands_hz
    ]
    if float(common["combined_max_samples"]) > min(per_band_budget):
        raise ValueError("common timewarp가 7-band 최엄격 timing budget을 넘습니다")
    return {
        "status": "PASS_STRUCTURAL_OFFLINE_ONLY",
        "map_sha256": map_sha,
        "knots_file_sha256": knots_ref["sha256"],
        "n_knots": int(adc.size),
        "combined_max_samples": float(common["combined_max_samples"]),
        "timing_budget_samples_by_subband": per_band_budget,
    }


def recompute_actual_err_coverage(
    source_aligned: np.ndarray,
    mics: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    segment_seconds: float = 1.5,
    segment_start_offset_seconds: float = 0.25,
    segment_count: int = 9,
) -> list[dict[str, Any]]:
    """실제 aligned source와 ERR ch0로 7-band segment metric을 재계산한다."""

    source = np.asarray(source_aligned, dtype=np.float64)
    microphones = np.asarray(mics)
    if source.ndim != 1 or microphones.ndim != 2 or microphones.shape[1] != 2:
        raise ValueError("source_aligned mono와 mics ERR/REF 2ch가 필요합니다")
    if source.shape[0] != SOURCE_FRAMES or microphones.shape[0] != SOURCE_FRAMES:
        raise ValueError("coverage 입력은 exact 15초 source/mics여야 합니다")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(microphones)):
        raise ValueError("coverage 입력에 NaN/Inf가 있습니다")
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError("recorded-v2 coverage sample rate는 48kHz여야 합니다")
    segment_frames = int(round(float(segment_seconds) * SAMPLE_RATE))
    first = int(round(float(segment_start_offset_seconds) * SAMPLE_RATE))
    starts = [first + index * segment_frames for index in range(int(segment_count))]
    if segment_frames != 72_000 or starts[-1] + segment_frames > SOURCE_FRAMES:
        raise ValueError("coverage deterministic 9×1.5초 population이 다릅니다")
    control = ControlBandContract.broadband_point_control()
    rows: list[dict[str, Any]] = []
    for start in starts:
        stop = start + segment_frames
        measured = measure_broadband_session(
            source[start:stop],
            np.asarray(microphones[start:stop, 0], dtype=np.float64),
            sample_rate=SAMPLE_RATE,
            subbands_hz=control.point_control_subbands_hz,
            nperseg=8192,
            noverlap=4096,
            min_coherence=MIN_SOURCE_ERR_COHERENCE,
            min_target_density=MIN_TARGET_D_DENSITY_RATIO,
        )
        rows.append(
            {
                "start_frame": start,
                "n_frames": segment_frames,
                "coherence": [float(value) for value in measured["coherence"]],
                "target_density_ratio": [
                    float(value) for value in measured["target_energy_density_ratio"]
                ],
                "joint_pass": [bool(value) for value in measured["joint_pass"]],
            }
        )
    return rows


def validate_stored_actual_err_coverage(
    stored: object,
    *,
    source_aligned: np.ndarray,
    mics: np.ndarray,
) -> dict[str, Any]:
    expected = recompute_actual_err_coverage(source_aligned, mics)
    if not isinstance(stored, list) or len(stored) != len(expected):
        raise ValueError("stored actual-ERR coverage row 수가 다릅니다")
    for index, (actual, recomputed) in enumerate(zip(stored, expected, strict=True)):
        row = _exact_keys(
            actual,
            {"start_frame", "n_frames", "coherence", "target_density_ratio", "joint_pass"},
            label=f"actual ERR coverage segment#{index}",
        )
        if row["start_frame"] != recomputed["start_frame"] or row["n_frames"] != recomputed["n_frames"]:
            raise ValueError("stored coverage segment population이 다릅니다")
        for label in ("coherence", "target_density_ratio"):
            left = np.asarray(row[label], dtype=np.float64)
            right = np.asarray(recomputed[label], dtype=np.float64)
            if left.shape != (7,) or not np.allclose(left, right, rtol=0.0, atol=1e-12):
                raise ValueError(f"stored {label}가 actual WAV 재계산과 다릅니다")
        if list(row["joint_pass"]) != recomputed["joint_pass"]:
            raise ValueError("stored joint_pass가 actual WAV 재계산과 다릅니다")
    all_band_rows = sum(all(row["joint_pass"]) for row in expected)
    return {
        "status": "PASS" if all_band_rows >= 8 else "BLOCKED",
        "segments": len(expected),
        "all_seven_band_joint_pass_segments": int(all_band_rows),
        "coverage_sha256": _sha256_bytes(_canonical_json(expected)),
    }


def publish_raw_capture_noreplace(
    *,
    repository_root: str | Path,
    target_directory: str | Path,
    submitted_output_pcm: np.ndarray,
    mics_raw_pcm: np.ndarray,
    callback_time_info: np.ndarray,
    receipt_metadata: Mapping[str, Any],
) -> Path:
    """analysis를 하지 않고 raw arrays+receipt만 sibling renameat2로 먼저 발행한다."""

    root = Path(repository_root).resolve(strict=True)
    target = Path(target_directory)
    target = target if target.is_absolute() else root / target
    target = Path(os.path.abspath(target))
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("raw capture target은 repository_root 내부여야 합니다") from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"raw capture를 덮어쓰지 않습니다: {target}")
    submitted_raw = np.asarray(submitted_output_pcm)
    microphones_raw = np.asarray(mics_raw_pcm)
    callback_raw = np.asarray(callback_time_info)
    if submitted_raw.dtype != np.dtype("<i2"):
        raise ValueError("submitted output은 capture 당시 little-endian int16이어야 합니다")
    if microphones_raw.dtype != np.dtype("<i4"):
        raise ValueError("raw mics는 capture 당시 little-endian int32여야 합니다")
    if callback_raw.dtype != np.dtype("<f8"):
        raise ValueError("callback witness는 float64 raw array여야 합니다")
    submitted = np.ascontiguousarray(submitted_raw)
    microphones = np.ascontiguousarray(microphones_raw)
    callback = np.ascontiguousarray(callback_raw)
    if submitted.ndim != 2 or submitted.shape[1] != 2 or submitted.shape[0] < SOURCE_FRAMES:
        raise ValueError("raw submitted output은 2ch이며 15초 이상이어야 합니다")
    if microphones.ndim != 2 or microphones.shape[1] != 2 or microphones.shape[0] != submitted.shape[0]:
        raise ValueError("raw mics는 submitted output과 같은 frame의 ERR/REF 2ch여야 합니다")
    if callback.ndim != 2 or callback.shape[1] != 6 or not np.all(np.isfinite(callback)):
        raise ValueError(
            "callback witness는 [adc_time,dac_time,frames,status,input_start,output_start] 6열이어야 합니다"
        )
    frames = callback[:, 2]
    status_flags = callback[:, 3]
    input_starts = callback[:, 4]
    output_starts = callback[:, 5]
    if (
        np.any(frames <= 0.0)
        or np.any(frames != np.rint(frames))
        or np.any(status_flags != np.rint(status_flags))
        or np.any(input_starts != np.rint(input_starts))
        or np.any(output_starts != np.rint(output_starts))
        or np.any(np.diff(callback[:, 0]) <= 0.0)
        or np.any(np.diff(callback[:, 1]) <= 0.0)
    ):
        raise ValueError("callback timestamp/frame/status/cursor witness가 유효하지 않습니다")
    expected_next = input_starts[:-1] + frames[:-1]
    observed_slips = int(
        np.count_nonzero(input_starts[1:] != expected_next)
        + np.count_nonzero(output_starts[1:] != output_starts[:-1] + frames[:-1])
        + np.count_nonzero(input_starts != output_starts)
    )
    observed_xruns = int(np.count_nonzero(status_flags))
    reserved = {
        "schema",
        "role",
        "raw_published_before_analysis",
        "analysis_started",
        "capture_valid_for_offline_analysis",
        "arrays",
        "evidence_sha256",
    }
    overlap = sorted(reserved & set(receipt_metadata))
    if overlap:
        raise ValueError(f"raw receipt metadata가 core field를 덮어쓸 수 없습니다: {overlap}")
    metadata = _exact_keys(
        dict(receipt_metadata),
        {
            "capture_id",
            "capture_contract_sha256",
            "source_plan",
            "source_plan_row_sha256",
            "plant_evidence",
            "sample_rate_hz",
            "block_size",
            "source_output_start_frame",
            "submitted_source_pcm_sha256",
            "submitted_source_stereo_pcm_sha256",
            "xrun_count",
            "clip_count",
            "sample_slip_count",
            "safety_confirmations",
            "device_occupancy_witness",
        },
        label="raw receipt metadata",
    )
    if not str(metadata["capture_id"]).strip() or str(metadata["capture_id"]) != target.name:
        raise ValueError("raw capture_id와 target directory name이 다릅니다")
    if metadata["capture_contract_sha256"] != capture_contract()["contract_sha256"]:
        raise ValueError("raw capture contract SHA가 현재 recorded-v2 계약과 다릅니다")
    if metadata["sample_rate_hz"] != SAMPLE_RATE or metadata["block_size"] != BLOCK_SIZE:
        raise ValueError("raw capture sample-rate/block이 48k/256이 아닙니다")
    source_start = metadata["source_output_start_frame"]
    if (
        isinstance(source_start, bool)
        or not isinstance(source_start, int)
        or source_start < 0
        or source_start + SOURCE_FRAMES > submitted.shape[0]
    ):
        raise ValueError("raw submitted output의 exact 15초 source region이 유효하지 않습니다")
    source_region = np.ascontiguousarray(
        submitted[source_start : source_start + SOURCE_FRAMES], dtype="<i2"
    )
    if np.any(source_region[:, 1] != 0):
        raise ValueError("recorded-v2 source region의 cancel output ch1은 exact zero여야 합니다")
    source_mono_sha = _sha256_bytes(source_region[:, 0].tobytes(order="C"))
    source_stereo_sha = _sha256_bytes(source_region.tobytes(order="C"))
    if source_mono_sha != _sha(
        metadata["submitted_source_pcm_sha256"], label="submitted source mono PCM SHA"
    ) or source_stereo_sha != _sha(
        metadata["submitted_source_stereo_pcm_sha256"],
        label="submitted source stereo PCM SHA",
    ):
        raise ValueError("raw exact 15초 submitted PCM SHA가 source plan과 다릅니다")
    _sha(metadata["source_plan_row_sha256"], label="source plan row SHA")
    _regular_file_reference(
        metadata["source_plan"], repository_root=root, label="raw source plan"
    )
    _regular_file_reference(
        metadata["plant_evidence"], repository_root=root, label="raw plant evidence"
    )
    confirmations = _exact_keys(
        metadata["safety_confirmations"],
        {"user_present", "volume_minimum", "routing_and_geometry"},
        label="raw safety confirmations",
    )
    if any(confirmations[key] is not True for key in confirmations):
        raise ValueError("raw capture의 사용자/볼륨/배선 확인이 모두 PASS가 아닙니다")
    _validate_device_occupancy_witness(
        metadata["device_occupancy_witness"], repository_root=root
    )
    for field in ("xrun_count", "clip_count", "sample_slip_count"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"raw {field}가 0 이상 정수가 아닙니다")
    observed_clip_count = int(
        np.count_nonzero(microphones == np.iinfo(np.int32).min)
        + np.count_nonzero(microphones == np.iinfo(np.int32).max)
    )
    if (
        metadata["xrun_count"] != observed_xruns
        or metadata["sample_slip_count"] != observed_slips
        or metadata["clip_count"] != observed_clip_count
    ):
        raise ValueError(
            "raw xrun/clip/sample-slip count가 callback/mics array 재계산과 다릅니다"
        )
    staging = target.parent / f".staging_{target.name}_{os.getpid()}_{os.urandom(8).hex()}"
    staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        arrays = {
            "submitted_output_int16.npy": submitted,
            "mics_raw_int32.npy": microphones,
            "callback_time_info_float64.npy": callback,
        }
        references: dict[str, Any] = {}
        for name, array in arrays.items():
            path = staging / name
            with path.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            references[name] = {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "array_sha256": _sha256_bytes(array.tobytes(order="C")),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        payload = {
            **metadata,
            "schema": RECORDED_V2_RAW_CAPTURE_SCHEMA,
            "role": "immutable_raw_before_analysis",
            "raw_published_before_analysis": True,
            "analysis_started": False,
            "capture_valid_for_offline_analysis": all(
                metadata[field] == 0
                for field in ("xrun_count", "clip_count", "sample_slip_count")
            ),
            "arrays": references,
        }
        payload["evidence_sha256"] = _sha256_bytes(_canonical_json(payload))
        receipt_path = staging / "raw_receipt.json"
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return publish_directory_noreplace(staging, target)
    except Exception:
        # 캡처자가 잡은 raw를 임의 삭제하지 않는다. staging은 실패 증거로 그대로 둔다.
        raise


def validate_raw_capture_bundle(
    raw_directory: str | Path,
    *,
    repository_root: str | Path,
    require_valid_for_analysis: bool = True,
) -> dict[str, Any]:
    """발행된 raw directory/NPY bytes/source region/safety lineage를 다시 검증한다."""

    root = Path(repository_root).resolve(strict=True)
    directory = Path(raw_directory)
    directory = directory if directory.is_absolute() else root / directory
    try:
        directory = directory.resolve(strict=True)
        directory.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("raw capture directory가 repository_root 안에 없습니다") from exc
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("raw capture는 symlink가 아닌 directory여야 합니다")
    receipt_path = directory / "raw_receipt.json"
    receipt = _sealed_payload(
        _json_no_duplicates(receipt_path),
        expected_schema=RECORDED_V2_RAW_CAPTURE_SCHEMA,
        label="recorded-v2 raw capture",
    )
    _exact_keys(
        receipt,
        {
            "capture_id",
            "capture_contract_sha256",
            "source_plan",
            "source_plan_row_sha256",
            "plant_evidence",
            "sample_rate_hz",
            "block_size",
            "source_output_start_frame",
            "submitted_source_pcm_sha256",
            "submitted_source_stereo_pcm_sha256",
            "xrun_count",
            "clip_count",
            "sample_slip_count",
            "safety_confirmations",
            "device_occupancy_witness",
            "schema",
            "role",
            "raw_published_before_analysis",
            "analysis_started",
            "capture_valid_for_offline_analysis",
            "arrays",
            "evidence_sha256",
        },
        label="recorded-v2 raw capture",
    )
    if (
        receipt["role"] != "immutable_raw_before_analysis"
        or receipt["capture_id"] != directory.name
        or receipt["raw_published_before_analysis"] is not True
        or receipt["analysis_started"] is not False
        or receipt["capture_contract_sha256"] != capture_contract()["contract_sha256"]
        or receipt["sample_rate_hz"] != SAMPLE_RATE
        or receipt["block_size"] != BLOCK_SIZE
    ):
        raise ValueError("raw-first/contract/sample-rate/block receipt가 다릅니다")
    _regular_file_reference(
        receipt["source_plan"], repository_root=root, label="raw source plan"
    )
    _regular_file_reference(
        receipt["plant_evidence"], repository_root=root, label="raw plant evidence"
    )
    _sha(receipt["source_plan_row_sha256"], label="raw source-plan row SHA")
    arrays = _exact_keys(
        receipt["arrays"],
        {
            "submitted_output_int16.npy",
            "mics_raw_int32.npy",
            "callback_time_info_float64.npy",
        },
        label="raw arrays",
    )
    loaded: dict[str, np.ndarray] = {}
    expected_dtypes = {
        "submitted_output_int16.npy": np.dtype("<i2"),
        "mics_raw_int32.npy": np.dtype("<i4"),
        "callback_time_info_float64.npy": np.dtype("<f8"),
    }
    for name, expected_dtype in expected_dtypes.items():
        reference = _exact_keys(
            arrays[name],
            {"path", "size_bytes", "sha256", "array_sha256", "shape", "dtype"},
            label=f"raw arrays.{name}",
        )
        if reference["path"] != name:
            raise ValueError("raw NPY path는 canonical basename이어야 합니다")
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"raw NPY regular file이 없습니다: {path}")
        if path.stat().st_size != reference["size_bytes"] or sha256_file(path) != _sha(
            reference["sha256"], label=f"{name} file SHA"
        ):
            raise ValueError(f"raw NPY file bytes가 receipt와 다릅니다: {name}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != expected_dtype or list(array.shape) != reference["shape"]:
            raise ValueError(f"raw NPY dtype/shape가 receipt와 다릅니다: {name}")
        if str(array.dtype) != reference["dtype"] or _sha256_bytes(
            np.ascontiguousarray(array).tobytes(order="C")
        ) != _sha(reference["array_sha256"], label=f"{name} array SHA"):
            raise ValueError(f"raw NPY array SHA가 receipt와 다릅니다: {name}")
        loaded[name] = array
    submitted = loaded["submitted_output_int16.npy"]
    microphones = loaded["mics_raw_int32.npy"]
    callback = loaded["callback_time_info_float64.npy"]
    if (
        submitted.ndim != 2
        or submitted.shape[1] != 2
        or microphones.shape != submitted.shape
        or callback.ndim != 2
        or callback.shape[1] != 6
    ):
        raise ValueError("raw array channel/frame 계약이 다릅니다")
    start = receipt["source_output_start_frame"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or start + SOURCE_FRAMES > submitted.shape[0]
    ):
        raise ValueError("raw source output region이 exact 15초가 아닙니다")
    region = np.ascontiguousarray(submitted[start : start + SOURCE_FRAMES], dtype="<i2")
    if np.any(region[:, 1] != 0):
        raise ValueError("raw source region cancel channel이 무음이 아닙니다")
    if _sha256_bytes(region[:, 0].tobytes(order="C")) != _sha(
        receipt["submitted_source_pcm_sha256"], label="raw source mono PCM SHA"
    ) or _sha256_bytes(region.tobytes(order="C")) != _sha(
        receipt["submitted_source_stereo_pcm_sha256"], label="raw source stereo PCM SHA"
    ):
        raise ValueError("raw source region PCM SHA가 plan lineage와 다릅니다")
    confirmations = _exact_keys(
        receipt["safety_confirmations"],
        {"user_present", "volume_minimum", "routing_and_geometry"},
        label="raw safety confirmations",
    )
    if any(value is not True for value in confirmations.values()):
        raise ValueError("raw live safety/device occupancy evidence가 PASS가 아닙니다")
    _validate_device_occupancy_witness(
        receipt["device_occupancy_witness"], repository_root=root
    )
    frames = callback[:, 2]
    status_flags = callback[:, 3]
    input_starts = callback[:, 4]
    output_starts = callback[:, 5]
    if (
        not np.all(np.isfinite(callback))
        or np.any(frames <= 0.0)
        or np.any(frames != np.rint(frames))
        or np.any(status_flags != np.rint(status_flags))
        or np.any(input_starts != np.rint(input_starts))
        or np.any(output_starts != np.rint(output_starts))
        or np.any(np.diff(callback[:, 0]) <= 0.0)
        or np.any(np.diff(callback[:, 1]) <= 0.0)
    ):
        raise ValueError("raw callback witness가 유효하지 않습니다")
    observed_slips = int(
        np.count_nonzero(input_starts[1:] != input_starts[:-1] + frames[:-1])
        + np.count_nonzero(output_starts[1:] != output_starts[:-1] + frames[:-1])
        + np.count_nonzero(input_starts != output_starts)
    )
    observed_xruns = int(np.count_nonzero(status_flags))
    observed_clips = int(
        np.count_nonzero(microphones == np.iinfo(np.int32).min)
        + np.count_nonzero(microphones == np.iinfo(np.int32).max)
    )
    if (
        receipt["xrun_count"] != observed_xruns
        or receipt["sample_slip_count"] != observed_slips
        or receipt["clip_count"] != observed_clips
    ):
        raise ValueError("raw stored xrun/clip/slip가 arrays 재계산과 다릅니다")
    counts_valid = all(
        isinstance(receipt[field], int)
        and not isinstance(receipt[field], bool)
        and receipt[field] == 0
        for field in ("xrun_count", "clip_count", "sample_slip_count")
    )
    if receipt["capture_valid_for_offline_analysis"] is not counts_valid:
        raise ValueError("raw capture validity boolean이 xrun/clip/slip 재계산과 다릅니다")
    if require_valid_for_analysis and not counts_valid:
        raise ValueError("xrun/clip/sample-slip raw는 보존하되 canonical 분석에 쓸 수 없습니다")
    return {
        "status": "PASS" if counts_valid else "INVALID_PRESERVED_RAW",
        "directory": directory,
        "receipt": receipt,
        "receipt_file_sha256": sha256_file(receipt_path),
        "raw_capture_sha256": receipt["evidence_sha256"],
        "submitted_output_pcm_sha256": arrays["submitted_output_int16.npy"]["array_sha256"],
        "mics_raw_pcm_sha256": arrays["mics_raw_int32.npy"]["array_sha256"],
        "submitted_output": submitted,
        "mics_raw": microphones,
        "callback_time_info": callback,
    }


def derive_aligned_session_arrays(
    *,
    raw_bundle: Mapping[str, Any],
    timewarp_receipt: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """absolute DAC-q grid에서 submitted source와 dewarped ERR/REF를 exact 15초로 만든다."""

    root = Path(repository_root).resolve(strict=True)
    warp = validate_timewarp_receipt(
        timewarp_receipt,
        repository_root=root,
        expected_raw_capture_sha256=str(raw_bundle["raw_capture_sha256"]),
        expected_submitted_pcm_sha256=str(raw_bundle["submitted_output_pcm_sha256"]),
        expected_mics_pcm_sha256=str(raw_bundle["mics_raw_pcm_sha256"]),
    )
    common = timewarp_receipt["common_map"]
    knots_ref = common["knots_file"]
    knots_path = root / str(knots_ref["path"])
    with np.load(knots_path, allow_pickle=False) as archive:
        adc_knots = np.asarray(archive["adc_frame_knots"], dtype=np.float64)
        dac_knots = np.asarray(archive["dac_q_knots"], dtype=np.float64)
    submitted = np.asarray(raw_bundle["submitted_output"], dtype="<i2")
    raw_mics = np.asarray(raw_bundle["mics_raw"], dtype="<i4")
    receipt = raw_bundle["receipt"]
    source_start = int(receipt["source_output_start_frame"])
    q_target = source_start + np.arange(SOURCE_FRAMES, dtype=np.float64)
    if q_target[0] < dac_knots[0] or q_target[-1] > dac_knots[-1]:
        raise ValueError("absolute DAC-q knots가 15초 source region 전체를 덮지 못합니다")
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import map_coordinates

    adc_at_q = CubicSpline(dac_knots, adc_knots, extrapolate=False)(q_target)
    if (
        not np.all(np.isfinite(adc_at_q))
        or np.min(adc_at_q) < 2.0
        or np.max(adc_at_q) > raw_mics.shape[0] - 3.0
        or not np.all(np.diff(adc_at_q) > 0.0)
    ):
        raise ValueError("inverse absolute DAC-q map이 raw ADC 범위/단조 조건을 벗어납니다")
    source = (
        submitted[source_start : source_start + SOURCE_FRAMES, 0].astype(np.float64)
        / 32768.0
    ).astype(np.float32)
    mics_float = raw_mics.astype(np.float64) / 2147483648.0
    aligned_channels = []
    coordinates = adc_at_q.reshape(1, -1)
    for channel in range(2):
        aligned_channels.append(
            map_coordinates(
                mics_float[:, channel],
                coordinates,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            ).astype(np.float32)
        )
    aligned_mics = np.column_stack(aligned_channels).astype(np.float32, copy=False)
    evidence = {
        "schema": "recorded_broadband_v2_aligned_arrays_v1",
        "method": "inverse_absolute_dac_q_cubic_spline_then_order3_adc_resample",
        "timewarp_map_sha256": warp["map_sha256"],
        "source_aligned_float32_sha256": _sha256_bytes(source.tobytes(order="C")),
        "mics_aligned_float32_sha256": _sha256_bytes(aligned_mics.tobytes(order="C")),
        "source_frames": SOURCE_FRAMES,
        "adc_position_min": float(np.min(adc_at_q)),
        "adc_position_max": float(np.max(adc_at_q)),
    }
    evidence["evidence_sha256"] = _sha256_bytes(_canonical_json(evidence))
    return source, aligned_mics, evidence


def publish_aligned_session_noreplace(
    *,
    repository_root: str | Path,
    target_directory: str | Path,
    raw_capture_directory: str | Path,
    timewarp_receipt_path: str | Path,
    session_identity: Mapping[str, Any],
) -> Path:
    """raw PASS 뒤 aligned WAV/7-band coverage/session을 no-replace 발행한다."""

    import soundfile as sf

    root = Path(repository_root).resolve(strict=True)
    target = Path(target_directory)
    target = target if target.is_absolute() else root / target
    target = Path(os.path.abspath(target))
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("aligned session target은 repository_root 내부여야 합니다") from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"aligned session을 덮어쓰지 않습니다: {target}")
    identity = _exact_keys(
        dict(session_identity),
        {
            "session_id",
            "split",
            "source_family",
            "group_id",
            "lineage_id",
            "source_plan_row_sha256",
            "native_source_sha256",
            "processed_source_sha256",
            "transform_receipt_sha256",
        },
        label="recorded-v2 session identity",
    )
    if (
        str(identity["session_id"]) != target.name
        or identity["split"] not in REQUIRED_SPLITS
        or identity["source_family"] not in REQUIRED_FAMILIES
        or not str(identity["group_id"]).strip()
        or not str(identity["lineage_id"]).strip()
    ):
        raise ValueError("session identity/path/split/family/group/lineage가 유효하지 않습니다")
    for field in (
        "source_plan_row_sha256",
        "native_source_sha256",
        "processed_source_sha256",
        "transform_receipt_sha256",
    ):
        _sha(identity[field], label=f"session identity {field}")
    raw = validate_raw_capture_bundle(
        raw_capture_directory,
        repository_root=root,
        require_valid_for_analysis=True,
    )
    if identity["source_plan_row_sha256"] != raw["receipt"]["source_plan_row_sha256"]:
        raise ValueError("session identity와 raw source-plan row SHA가 다릅니다")
    warp_path = Path(timewarp_receipt_path)
    warp_path = warp_path if warp_path.is_absolute() else root / warp_path
    warp_path = warp_path.resolve(strict=True)
    warp_path.relative_to(root)
    warp_receipt = _json_no_duplicates(warp_path)
    source, aligned_mics, alignment = derive_aligned_session_arrays(
        raw_bundle=raw,
        timewarp_receipt=warp_receipt,
        repository_root=root,
    )
    staging = target.parent / f".staging_{target.name}_{os.getpid()}_{os.urandom(8).hex()}"
    staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        sf.write(staging / "source.wav", source, SAMPLE_RATE, subtype="FLOAT")
        sf.write(staging / "source_aligned.wav", source, SAMPLE_RATE, subtype="FLOAT")
        sf.write(staging / "mics.wav", aligned_mics, SAMPLE_RATE, subtype="PCM_32")
        persisted_source, source_rate = sf.read(
            staging / "source_aligned.wav", dtype="float32", always_2d=False
        )
        persisted_mics, mics_rate = sf.read(
            staging / "mics.wav", dtype="float32", always_2d=True
        )
        if source_rate != SAMPLE_RATE or mics_rate != SAMPLE_RATE:
            raise ValueError("persisted aligned WAV sample rate가 48kHz가 아닙니다")
        # PCM_32 write/read 양자화를 포함한 **실제 persisted arrays**가 후속 coverage의
        # 입력이다. 메모리상의 pre-write float SHA를 authority로 남기지 않는다.
        alignment = {
            key: value for key, value in alignment.items() if key != "evidence_sha256"
        }
        alignment["source_aligned_float32_sha256"] = _sha256_bytes(
            np.asarray(persisted_source, dtype=np.float32).tobytes(order="C")
        )
        alignment["mics_aligned_float32_sha256"] = _sha256_bytes(
            np.asarray(persisted_mics, dtype=np.float32).tobytes(order="C")
        )
        alignment["evidence_sha256"] = _sha256_bytes(_canonical_json(alignment))
        coverage_rows = recompute_actual_err_coverage(persisted_source, persisted_mics)
        coverage_summary = validate_stored_actual_err_coverage(
            coverage_rows,
            source_aligned=persisted_source,
            mics=persisted_mics,
        )
        if coverage_summary["status"] != "PASS":
            raise ValueError(
                "actual ERR에서 all-seven-band PASS segment가 8개 미만입니다; "
                "raw는 보존하지만 canonical session은 발행하지 않습니다"
            )
        coverage = {
            "schema": "recorded_broadband_v2_actual_err_coverage_v1",
            "role": "recomputed_from_persisted_source_aligned_and_mics_err_ch0",
            "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
            "segments": coverage_rows,
            "summary": coverage_summary,
        }
        coverage["evidence_sha256"] = _sha256_bytes(_canonical_json(coverage))
        with (staging / "coverage.json").open("x", encoding="utf-8") as handle:
            json.dump(coverage, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        artifacts = {}
        for name in ("source.wav", "source_aligned.wav", "mics.wav", "coverage.json"):
            path = staging / name
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts[name] = {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        session = {
            "schema": RECORDED_V2_SESSION_SCHEMA,
            "role": "canonical_aligned_after_immutable_raw",
            **identity,
            "capture_contract_sha256": capture_contract()["contract_sha256"],
            "raw_capture": file_reference(
                raw["directory"] / "raw_receipt.json", repository_root=root
            ),
            "timewarp_receipt": file_reference(warp_path, repository_root=root),
            "alignment": alignment,
            "artifacts": artifacts,
            "coverage_evidence_sha256": coverage["evidence_sha256"],
        }
        session["evidence_sha256"] = _sha256_bytes(_canonical_json(session))
        with (staging / "session.json").open("x", encoding="utf-8") as handle:
            json.dump(session, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return publish_directory_noreplace(staging, target)
    except Exception:
        # canonical target에는 들어가지 않는다. raw bundle은 이미 별도 immutable 경로에 있다.
        raise


__all__ = [
    "BLOCK_SIZE",
    "CLOCK_FIT_BAND_HZ",
    "FULLBAND_CAUSAL_PLANT_SCHEMA",
    "MAX_SUBMITTED_PEAK_INT16",
    "RECORDED_V2_CAPTURE_CONTRACT_SCHEMA",
    "RECORDED_V2_CAMPAIGN_SELECTION_SCHEMA",
    "RECORDED_V2_CANDIDATE_SET_SCHEMA",
    "RECORDED_V2_LIVE_AUTHORITY",
    "RECORDED_V2_RAW_CAPTURE_SCHEMA",
    "RECORDED_V2_SESSION_SCHEMA",
    "RECORDED_V2_SOURCE_PLAN_SCHEMA",
    "RECORDED_V2_TIMEWARP_SCHEMA",
    "RecordedV2Blocked",
    "SAMPLE_RATE",
    "SOURCE_FRAMES",
    "SOURCE_SECONDS",
    "capture_contract",
    "derive_aligned_session_arrays",
    "file_reference",
    "publish_aligned_session_noreplace",
    "publish_raw_capture_noreplace",
    "recompute_actual_err_coverage",
    "render_submitted_pcm",
    "sha256_file",
    "submitted_pcm_evidence",
    "validate_fullband_causal_plant",
    "validate_raw_capture_bundle",
    "validate_source_plan",
    "validate_stored_actual_err_coverage",
    "validate_timewarp_receipt",
]
