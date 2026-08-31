"""Stage-2 2 kHz same-capture P/S를 tensor operator로 바꾸는 typed 경계.

이 loader는 JSON의 ``PASS`` 문구만 믿지 않고 P/S NPZ, raw, analysis, level,
relative-clock receipt의 실제 bytes SHA를 다시 계산한다. P/S NPZ의 대역·반복·
clock/timing·callback 필드도 Stage-2 계약과 다시 비교한다. lead integer를
받지 않고 오직 :class:`PlantDelays`와 :class:`TrainingTimingContract`로 유도한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from ..dsp.stage2_2khz_level_contract import (
    MINIMUM_ACTUATOR_HEADROOM_DB,
    canonical_stage2_operating_level_contract,
)
from ..dsp.timing import PlantDelays, TrainingTimingContract
from ..losses.broadband_loss import CausalFIRPathData
from .stage2_2khz_git_authority import (
    verify_source_commit_ancestor,
    verify_tracked_head_authority,
)


STAGE2_2KHZ_PATH_NPZ_SCHEMA = "stage2_2khz_causal_path_npz_v1"
STAGE2_2KHZ_PLANT_BINDING_SCHEMA = "stage2_2khz_plant_binding_v1"
STAGE2_2KHZ_RELATIVE_CLOCK_MODEL_SCHEMA = "stage2_2khz_relative_clock_model_v1"
STAGE2_2KHZ_RAW_CAPTURE_SCHEMA = "stage2_2khz_raw_capture_npz_v1"
STAGE2_2KHZ_ANALYSIS_SCHEMA = "stage2_2khz_analysis_npz_v1"
STAGE2_2KHZ_PHYSICAL_AUTHORITY_SCHEMA = "stage2_2khz_physical_git_authority_v1"
STAGE2_2KHZ_PHYSICAL_AUTHORITY_PATH = "authority/stage2_2khz_physical.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_PATH_KEYS = frozenset(
    {
        "stage2_path_schema",
        "measurement_status",
        "canonical_training_eligible",
        "stage2_contract_id",
        "stage2_contract_sha256",
        "role",
        "fir",
        "delay_samples",
        "fractional_delay_samples",
        "delay_semantics",
        "sample_rate",
        "capture_id",
        "excitation_band_hz",
        "band_consistency_hz",
        "band_consistency",
        "independent_epoch_role_names",
        "independent_epoch_start_frames",
        "independent_epoch_stop_frames",
        "independent_epoch_kept",
        "repeated_slot_count",
        "timing_residual_max_samples",
        "xrun_count",
        "clip_count",
        "sample_slip_count",
        "callback_status_failures",
        "output_pcm_provenance",
        "source_raw_npz_path",
        "source_raw_npz_sha256",
        "source_analysis_npz_path",
        "source_analysis_npz_sha256",
        "calibration_block_size",
        "error_mic_channel",
        "reference_mic_channel",
        "source_capture_commit_sha",
    }
)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError(f"artifact snapshot 중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"artifact symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"artifact byte 크기가 snapshot과 다릅니다: {path}")
    return content, hashlib.sha256(content).hexdigest()


def _inside_repository(root: Path, raw: object, *, label: str) -> Path:
    relative = Path(str(raw or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            node = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    return root / relative


def _artifact_ref(root: Path, value: object, *, label: str) -> tuple[Path, bytes, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label}는 exact path/sha256 mapping이어야 합니다")
    expected = _require_sha256(value["sha256"], label=f"{label}.sha256")
    path = _inside_repository(root, value["path"], label=label)
    try:
        content, actual = _snapshot_regular_file(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} artifact가 없습니다: {path}") from exc
    if actual != expected:
        raise ValueError(f"{label} bytes SHA가 binding과 다릅니다")
    return path, content, actual


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"Stage-2 path {key}는 scalar여야 합니다")
    return value.item()


def _exact_bands(value: np.ndarray, expected: tuple[tuple[float, float], ...]) -> bool:
    parsed = np.asarray(value, dtype=np.float64)
    return parsed.shape == (len(expected), 2) and np.array_equal(
        parsed, np.asarray(expected, dtype=np.float64)
    )


@dataclass(frozen=True)
class Stage2SourceOperatingLevelBinding:
    """physical source/control level과 P/S actuator feasibility의 typed 결과."""

    planned_contract_sha256: str
    physical_evidence_file_sha256: str
    physical_evidence_payload_sha256: str
    source_operating_peak_abs: float
    actuator_limit_abs: float
    augmentation_gain_db_minimum: float
    augmentation_gain_db_maximum: float
    post_gain_hard_peak_cap_abs: float
    minimum_observed_actuator_headroom_db: float
    broadband_required_control_peak_upper_bound_abs: float
    actuator_feasibility_passed: bool
    fixture_only: bool = False

    def __post_init__(self) -> None:
        planned = canonical_stage2_operating_level_contract()
        gain = planned["augmentation_gain_db"]
        expected = {
            "planned_contract_sha256": planned["canonical_payload_sha256"],
            "source_operating_peak_abs": float(planned["source_operating_peak_abs"]),
            "actuator_limit_abs": float(planned["actuator_limit_abs"]),
            "augmentation_gain_db_minimum": float(gain["minimum"]),
            "augmentation_gain_db_maximum": float(gain["maximum"]),
            "post_gain_hard_peak_cap_abs": float(
                gain["post_gain_hard_peak_cap_abs"]
            ),
        }
        for key, value in expected.items():
            actual = getattr(self, key)
            if isinstance(value, float):
                if not math.isclose(
                    float(actual), value, rel_tol=0.0, abs_tol=1.0e-15
                ):
                    raise ValueError(f"Stage-2 source operating level {key}가 canonical과 다릅니다")
            elif actual != value:
                raise ValueError(f"Stage-2 source operating level {key}가 canonical과 다릅니다")
        for label, value in (
            ("physical evidence file", self.physical_evidence_file_sha256),
            ("physical evidence payload", self.physical_evidence_payload_sha256),
        ):
            _require_sha256(value, label=f"Stage-2 {label} SHA")
        headroom = float(self.minimum_observed_actuator_headroom_db)
        required = float(self.broadband_required_control_peak_upper_bound_abs)
        if (
            self.actuator_feasibility_passed is not True
            or not math.isfinite(headroom)
            or headroom < MINIMUM_ACTUATOR_HEADROOM_DB
            or not math.isfinite(required)
            or required <= 0.0
        ):
            raise ValueError("Stage-2 actual P/S 3 dB actuator feasibility가 PASS가 아닙니다")
        recomputed_headroom = 20.0 * math.log10(
            float(self.actuator_limit_abs) / required
        )
        if recomputed_headroom + 1.0e-12 < MINIMUM_ACTUATOR_HEADROOM_DB:
            raise ValueError("Stage-2 broadband actuator/limiter headroom이 3 dB 미만입니다")

    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "planned_contract_sha256": self.planned_contract_sha256,
                    "physical_evidence_file_sha256": self.physical_evidence_file_sha256,
                    "physical_evidence_payload_sha256": self.physical_evidence_payload_sha256,
                    "source_operating_peak_abs": self.source_operating_peak_abs,
                    "actuator_limit_abs": self.actuator_limit_abs,
                    "augmentation_gain_db_minimum": self.augmentation_gain_db_minimum,
                    "augmentation_gain_db_maximum": self.augmentation_gain_db_maximum,
                    "post_gain_hard_peak_cap_abs": self.post_gain_hard_peak_cap_abs,
                    "minimum_observed_actuator_headroom_db": self.minimum_observed_actuator_headroom_db,
                    "broadband_required_control_peak_upper_bound_abs": self.broadband_required_control_peak_upper_bound_abs,
                    "actuator_feasibility_passed": self.actuator_feasibility_passed,
                    "fixture_only": self.fixture_only,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class Stage2TwoKilohertzPlantBinding:
    """production loader가 실제 bytes를 재검증한 in-memory binding."""

    control_band_contract: Stage2TwoKilohertzContract
    control_band_contract_sha256: str
    training_timing_contract: TrainingTimingContract
    training_timing_contract_sha256: str
    primary_operator: CausalFIRPathData
    secondary_operator: CausalFIRPathData
    primary_path_sha256: str
    secondary_path_sha256: str
    raw_capture_sha256: str
    analysis_sha256: str
    measurement_level_evidence_sha256: str
    relative_clock_model_receipt_sha256: str
    verified_physical_subbands_hz: tuple[tuple[float, float], ...]
    err_channel_index: int
    reference_channel_index: int
    block_size: int
    binding_file_sha256: str
    source_capture_commit_sha: str
    source_operating_level: Stage2SourceOperatingLevelBinding | None = None
    fixture_only: bool = False
    schema_version: str = STAGE2_2KHZ_PLANT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        canonical = Stage2TwoKilohertzContract.canonical()
        if self.schema_version != STAGE2_2KHZ_PLANT_BINDING_SCHEMA:
            raise ValueError("Stage-2 plant binding schema가 다릅니다")
        if self.control_band_contract != canonical:
            raise ValueError("Stage-2 plant binding contract payload가 canonical과 다릅니다")
        if self.control_band_contract_sha256 != canonical.digest():
            raise ValueError("Stage-2 plant binding contract SHA가 다릅니다")
        if self.training_timing_contract_sha256 != self.training_timing_contract.digest():
            raise ValueError("Stage-2 TrainingTimingContract payload/SHA가 다릅니다")
        if int(self.block_size) != 256:
            raise ValueError("Stage-2 callback block은 256 samples여야 합니다")
        if self.primary_operator.role != "primary" or self.secondary_operator.role != "secondary":
            raise ValueError("Stage-2 P/S operator role이 뒤바뀌었습니다")
        timing = self.training_timing_contract
        if int(self.primary_operator.coarse_delay_samples) != int(
            timing.primary_zeros_before_fir_samples
        ):
            raise ValueError("Stage-2 P delay와 timing-v2가 다릅니다")
        if int(self.secondary_operator.coarse_delay_samples) != int(
            timing.secondary_delay_samples
        ):
            raise ValueError("Stage-2 S delay와 timing-v2가 다릅니다")
        if int(self.secondary_operator.handoff_extra_samples) != int(
            timing.handoff_samples
        ):
            raise ValueError("Stage-2 S handoff와 timing-v2가 다릅니다")
        expected = tuple(canonical.physical_identification_subbands_hz)
        if tuple(self.verified_physical_subbands_hz) != expected:
            raise ValueError("Stage-2 verified physical subband 6개가 exact하지 않습니다")
        for label, value in (
            ("P path", self.primary_path_sha256),
            ("S path", self.secondary_path_sha256),
            ("raw capture", self.raw_capture_sha256),
            ("analysis", self.analysis_sha256),
            ("measurement level", self.measurement_level_evidence_sha256),
            ("relative clock model", self.relative_clock_model_receipt_sha256),
            ("binding file", self.binding_file_sha256),
        ):
            _require_sha256(value, label=f"Stage-2 {label} SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_capture_commit_sha):
            raise ValueError("Stage-2 source capture commit은 40자리 SHA여야 합니다")
        if self.source_operating_level is not None:
            if not isinstance(
                self.source_operating_level, Stage2SourceOperatingLevelBinding
            ):
                raise TypeError("Stage-2 source operating level typed binding이 필요합니다")
            if not self.fixture_only and self.source_operating_level.fixture_only:
                raise ValueError("production P/S가 fixture source operating level을 소비합니다")

    @property
    def required_prefix_samples(self) -> int:
        return max(
            int(self.primary_operator.history_samples),
            int(self.secondary_operator.history_samples),
        )

    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "schema": self.schema_version,
                    "contract_sha256": self.control_band_contract_sha256,
                    "timing_sha256": self.training_timing_contract_sha256,
                    "primary_path_sha256": self.primary_path_sha256,
                    "secondary_path_sha256": self.secondary_path_sha256,
                    "raw_capture_sha256": self.raw_capture_sha256,
                    "analysis_sha256": self.analysis_sha256,
                    "measurement_level_evidence_sha256": self.measurement_level_evidence_sha256,
                    "relative_clock_model_receipt_sha256": self.relative_clock_model_receipt_sha256,
                    "binding_file_sha256": self.binding_file_sha256,
                    "source_capture_commit_sha": self.source_capture_commit_sha,
                    "source_operating_level_sha256": (
                        self.source_operating_level.digest()
                        if self.source_operating_level is not None
                        else None
                    ),
                    "fixture_only": bool(self.fixture_only),
                }
            )
        ).hexdigest()


def _validate_path_npz(
    content: bytes,
    *,
    role: Literal["primary", "secondary"],
    file_sha256: str,
    binding_sha256: str,
    raw_path: str,
    raw_sha256: str,
    analysis_path: str,
    analysis_sha256: str,
) -> tuple[CausalFIRPathData, dict[str, Any]]:
    contract = Stage2TwoKilohertzContract.canonical()
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            missing = _REQUIRED_PATH_KEYS - set(archive.files)
            if missing:
                raise ValueError(f"Stage-2 {role} NPZ 필드가 누락됐습니다: {sorted(missing)}")
            values = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Stage-2"):
            raise
        raise ValueError(f"Stage-2 {role} NPZ를 열 수 없습니다") from exc

    exact_scalars = {
        "stage2_path_schema": STAGE2_2KHZ_PATH_NPZ_SCHEMA,
        "measurement_status": "PASS",
        "canonical_training_eligible": True,
        "stage2_contract_id": contract.contract_id,
        "stage2_contract_sha256": contract.digest(),
        "role": role,
        "delay_semantics": "effective_zeros_before_compact_fir",
        "sample_rate": 48_000,
        "calibration_block_size": 256,
        "output_pcm_provenance": "observed_submitted_int16",
        "source_raw_npz_path": raw_path,
        "source_raw_npz_sha256": raw_sha256,
        "source_analysis_npz_path": analysis_path,
        "source_analysis_npz_sha256": analysis_sha256,
        "source_capture_commit_sha": None,
    }
    source_capture_commit_sha = str(_scalar(values, "source_capture_commit_sha"))
    if not re.fullmatch(r"[0-9a-f]{40}", source_capture_commit_sha):
        raise ValueError(f"Stage-2 {role} source capture commit이 40자리 SHA가 아닙니다")
    exact_scalars["source_capture_commit_sha"] = source_capture_commit_sha
    for key, expected in exact_scalars.items():
        if _scalar(values, key) != expected:
            raise ValueError(f"Stage-2 {role} NPZ {key}가 binding/계약과 다릅니다")
    for key in ("xrun_count", "clip_count", "sample_slip_count", "callback_status_failures"):
        if int(_scalar(values, key)) != 0:
            raise ValueError(f"Stage-2 {role} NPZ {key}가 0이 아닙니다")
    excitation = np.asarray(values["excitation_band_hz"], dtype=np.float64)
    if excitation.shape != (2,) or not np.array_equal(
        excitation,
        np.asarray(
            [contract.required_excitation_lower_hz, contract.required_excitation_upper_hz],
            dtype=np.float64,
        ),
    ):
        raise ValueError(f"Stage-2 {role} actual excitation은 exact [80,2828.427...]Hz여야 합니다")
    if not _exact_bands(
        values["band_consistency_hz"],
        tuple(contract.physical_identification_subbands_hz),
    ):
        raise ValueError(f"Stage-2 {role} consistency subband 6개가 exact하지 않습니다")
    consistency = np.asarray(values["band_consistency"], dtype=np.float64).reshape(-1)
    if (
        consistency.shape != (6,)
        or not np.all(np.isfinite(consistency))
        or np.any(consistency < 0.95)
    ):
        raise ValueError(f"Stage-2 {role} subband consistency가 0.95 미만입니다")
    if int(_scalar(values, "repeated_slot_count")) != 0:
        raise ValueError(f"Stage-2 {role} aperiodic capture에 periodic repeat slot을 선언할 수 없습니다")
    epoch_roles = tuple(str(value) for value in np.asarray(values["independent_epoch_role_names"]).reshape(-1))
    epoch_starts = np.asarray(values["independent_epoch_start_frames"], dtype=np.int64).reshape(-1)
    epoch_stops = np.asarray(values["independent_epoch_stop_frames"], dtype=np.int64).reshape(-1)
    epoch_kept = np.asarray(values["independent_epoch_kept"])
    if not (
        len(epoch_roles)
        == epoch_starts.size
        == epoch_stops.size
        == epoch_kept.size
        and epoch_kept.dtype == np.dtype("bool")
    ):
        raise ValueError(f"Stage-2 {role} independent epoch evidence shape/dtype가 다릅니다")
    previous_stop = -1
    kept_per_role = {name: 0 for name in ("fit_a", "fit_b", "untouched_holdout")}
    for epoch_role, start, stop, kept_flag in zip(
        epoch_roles, epoch_starts, epoch_stops, epoch_kept, strict=True
    ):
        if (
            epoch_role not in kept_per_role
            or int(start) < previous_stop
            or int(stop) <= int(start)
            or int(stop) - int(start) != 48_000
        ):
            raise ValueError(f"Stage-2 {role} independent epoch가 1s 비중첩 규약을 위반합니다")
        previous_stop = int(stop)
        if bool(kept_flag):
            kept_per_role[epoch_role] += 1
    if any(count < 8 for count in kept_per_role.values()):
        raise ValueError(f"Stage-2 {role} independent kept epoch가 role당 8개 미만입니다")
    residual = float(_scalar(values, "timing_residual_max_samples"))
    if not math.isfinite(residual) or residual > 0.270208:
        raise ValueError(f"Stage-2 {role} timing residual이 0.270208 samples를 넘습니다")
    fractional = float(_scalar(values, "fractional_delay_samples"))
    delay = int(_scalar(values, "delay_samples"))
    if delay < 0 or not math.isfinite(fractional) or not -0.5 <= fractional < 0.5:
        raise ValueError(f"Stage-2 {role} delay/fractional delay가 잘못됐습니다")
    fir = np.asarray(values["fir"])
    if fir.dtype != np.dtype("<f8") and fir.dtype != np.dtype("=f8"):
        raise ValueError(f"Stage-2 {role} FIR은 float64 bytes여야 합니다")
    fir = np.ascontiguousarray(fir, dtype=np.float64).reshape(-1)
    if fir.size < 1 or not np.all(np.isfinite(fir)) or float(np.max(np.abs(fir))) <= 0.0:
        raise ValueError(f"Stage-2 {role} FIR이 finite/nonzero가 아닙니다")
    fir_sha = hashlib.sha256(fir.tobytes(order="C")).hexdigest()
    capture_id = str(_scalar(values, "capture_id"))
    if not capture_id:
        raise ValueError(f"Stage-2 {role} capture_id가 비었습니다")
    internal_sha = hashlib.sha256(
        _canonical_json(
            {
                "schema": STAGE2_2KHZ_PATH_NPZ_SCHEMA,
                "role": role,
                "contract_sha256": contract.digest(),
                "capture_id": capture_id,
                "delay_samples": delay,
                "fractional_delay_samples": fractional,
                "fir_sha256": fir_sha,
            }
        )
    ).hexdigest()
    operator = CausalFIRPathData(
        role=role,
        post_onset_fir=fir,
        coarse_delay_samples=delay,
        fractional_delay_samples=fractional,
        support_samples=int(fir.size),
        sample_rate=48_000,
        handoff_extra_samples=0 if role == "primary" else 256,
        operator_file_sha256=file_sha256,
        operator_internal_sha256=internal_sha,
        fir_sha256=fir_sha,
        authority_sha256=binding_sha256,
        source_path=str(_scalar(values, "source_analysis_npz_path")),
        fractional_delay_encoded_in_post_onset_fir=True,
    )
    return operator, {
        "capture_id": capture_id,
        "err_channel_index": int(_scalar(values, "error_mic_channel")),
        "reference_channel_index": int(_scalar(values, "reference_mic_channel")),
        "delay_samples": delay,
        "band_consistency": tuple(float(value) for value in consistency),
        "timing_residual_max_samples": residual,
        "source_capture_commit_sha": source_capture_commit_sha,
    }


def _load_npz_bytes(content: bytes, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}는 allow_pickle=False NPZ여야 합니다") from exc


def _validate_raw_analysis_bytes(
    raw_content: bytes,
    analysis_content: bytes,
    *,
    raw_sha256: str,
) -> dict[str, Any]:
    """raw PCM과 분석 결과를 실제 array bytes에서 다시 결속한다.

    전기 absolute-frame clock이 존재한다고 가정하지 않는다. raw의 submitted stereo
    known-code와 acoustic capture가 analysis의 같은 capture ID/raw SHA로 이어지는지만
    검증하고, 상대 shared-q clock model은 별도 receipt가 담당한다.
    """

    contract = Stage2TwoKilohertzContract.canonical()
    raw = _load_npz_bytes(raw_content, label="Stage-2 raw capture")
    analysis = _load_npz_bytes(analysis_content, label="Stage-2 analysis")
    raw_required = {
        "stage2_raw_schema",
        "stage2_contract_sha256",
        "capture_id",
        "sample_rate",
        "block_size",
        "submitted_output_pcm",
        "captured_input_pcm",
        "xrun_count",
        "clip_count",
        "sample_slip_count",
        "callback_status_failures",
        "source_capture_commit_sha",
    }
    analysis_required = {
        "stage2_analysis_schema",
        "analysis_status",
        "stage2_contract_sha256",
        "capture_id",
        "raw_capture_sha256",
        "sample_rate",
        "physical_subbands_hz",
        "primary_band_consistency",
        "secondary_band_consistency",
        "primary_delay_samples",
        "secondary_delay_samples",
        "timing_residual_max_samples",
        "submitted_output_pcm_sha256",
        "source_capture_commit_sha",
    }
    if not raw_required.issubset(raw):
        raise ValueError(
            f"Stage-2 raw capture 필드가 누락됐습니다: {sorted(raw_required - set(raw))}"
        )
    if not analysis_required.issubset(analysis):
        raise ValueError(
            "Stage-2 analysis 필드가 누락됐습니다: "
            f"{sorted(analysis_required - set(analysis))}"
        )
    if (
        _scalar(raw, "stage2_raw_schema") != STAGE2_2KHZ_RAW_CAPTURE_SCHEMA
        or _scalar(raw, "stage2_contract_sha256") != contract.digest()
        or int(_scalar(raw, "sample_rate")) != 48_000
        or int(_scalar(raw, "block_size")) != 256
    ):
        raise ValueError("Stage-2 raw capture schema/contract/rate/block이 다릅니다")
    for key in ("xrun_count", "clip_count", "sample_slip_count", "callback_status_failures"):
        if int(_scalar(raw, key)) != 0:
            raise ValueError(f"Stage-2 raw capture {key}가 0이 아닙니다")
    submitted = np.asarray(raw["submitted_output_pcm"])
    captured = np.asarray(raw["captured_input_pcm"])
    if (
        submitted.dtype != np.dtype("int16")
        or submitted.ndim != 2
        or submitted.shape[1] != 2
        or submitted.shape[0] < 256
        or captured.dtype != np.dtype("int32")
        or captured.shape != submitted.shape
    ):
        raise ValueError("Stage-2 raw actual submitted/captured PCM dtype/shape가 다릅니다")
    submitted_sha = hashlib.sha256(
        np.ascontiguousarray(submitted).tobytes(order="C")
    ).hexdigest()
    capture_id = str(_scalar(raw, "capture_id"))
    source_capture_commit_sha = str(_scalar(raw, "source_capture_commit_sha"))
    if not capture_id:
        raise ValueError("Stage-2 raw capture_id가 비었습니다")
    if not re.fullmatch(r"[0-9a-f]{40}", source_capture_commit_sha):
        raise ValueError("Stage-2 raw source capture commit이 40자리 SHA가 아닙니다")
    if (
        _scalar(analysis, "stage2_analysis_schema") != STAGE2_2KHZ_ANALYSIS_SCHEMA
        or _scalar(analysis, "analysis_status") != "PASS"
        or _scalar(analysis, "stage2_contract_sha256") != contract.digest()
        or _scalar(analysis, "capture_id") != capture_id
        or _scalar(analysis, "raw_capture_sha256") != raw_sha256
        or int(_scalar(analysis, "sample_rate")) != 48_000
        or _scalar(analysis, "submitted_output_pcm_sha256") != submitted_sha
        or _scalar(analysis, "source_capture_commit_sha") != source_capture_commit_sha
    ):
        raise ValueError("Stage-2 analysis가 실제 raw/capture/submitted PCM bytes와 다릅니다")
    if not _exact_bands(
        analysis["physical_subbands_hz"],
        tuple(contract.physical_identification_subbands_hz),
    ):
        raise ValueError("Stage-2 analysis physical subband 6개가 exact하지 않습니다")
    primary_consistency = np.asarray(
        analysis["primary_band_consistency"], dtype=np.float64
    ).reshape(-1)
    secondary_consistency = np.asarray(
        analysis["secondary_band_consistency"], dtype=np.float64
    ).reshape(-1)
    if (
        primary_consistency.shape != (6,)
        or secondary_consistency.shape != (6,)
        or not np.all(np.isfinite(primary_consistency))
        or not np.all(np.isfinite(secondary_consistency))
        or np.any(primary_consistency < 0.95)
        or np.any(secondary_consistency < 0.95)
    ):
        raise ValueError("Stage-2 analysis P/S subband consistency가 0.95 미만입니다")
    residual = float(_scalar(analysis, "timing_residual_max_samples"))
    if not math.isfinite(residual) or residual > 0.270208:
        raise ValueError("Stage-2 analysis timing residual이 0.270208 samples를 넘습니다")
    primary_delay = int(_scalar(analysis, "primary_delay_samples"))
    secondary_delay = int(_scalar(analysis, "secondary_delay_samples"))
    if primary_delay < 0 or secondary_delay < 0:
        raise ValueError("Stage-2 analysis P/S delay가 음수입니다")
    return {
        "capture_id": capture_id,
        "submitted_output_pcm_sha256": submitted_sha,
        "primary_band_consistency": tuple(float(value) for value in primary_consistency),
        "secondary_band_consistency": tuple(float(value) for value in secondary_consistency),
        "primary_delay_samples": primary_delay,
        "secondary_delay_samples": secondary_delay,
        "timing_residual_max_samples": residual,
        "source_capture_commit_sha": source_capture_commit_sha,
    }


def _load_self_attested_stage2_2khz_plant_binding_for_test(
    binding_path: str | Path,
    *,
    repository_root: str | Path,
    expected_binding_file_sha256: str | None = None,
) -> Stage2TwoKilohertzPlantBinding:
    """자가서명 bytes의 의미 검사용 parser.

    이 함수의 결과는 언제나 ``fixture_only=True``이며 production admission에 사용할
    수 없다. 파일 내부의 PASS/fixture_only=false는 외부 물리 권한이 아니므로, 테스트가
    raw/analysis 재계산을 검증할 때만 직접 사용한다.
    """

    root = Path(repository_root).resolve(strict=True)
    candidate = Path(binding_path)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError("Stage-2 binding은 repository 내부에 있어야 합니다") from exc
    else:
        relative = candidate
    target = _inside_repository(root, relative.as_posix(), label="Stage-2 plant binding")
    content, binding_sha = _snapshot_regular_file(target)
    if expected_binding_file_sha256 is not None and binding_sha != _require_sha256(
        expected_binding_file_sha256, label="expected Stage-2 binding SHA"
    ):
        raise ValueError("Stage-2 binding bytes SHA가 profile과 다릅니다")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 plant binding은 UTF-8 JSON이어야 합니다") from exc
    expected_keys = {
        "schema",
        "status",
        "canonical_training_eligible",
        "fixture_only",
        "control_band_contract",
        "sample_rate_hz",
        "block_size",
        "verified_physical_subbands_hz",
        "minimum_subband_consistency",
        "maximum_timing_residual_samples",
        "minimum_independent_epochs_per_role",
        "periodic_repeat_indices_allowed",
        "handoff_extra_samples",
        "lead_source",
        "primary_path",
        "secondary_path",
        "raw_capture",
        "analysis",
        "measurement_level_evidence",
        "relative_clock_model_receipt",
        "err_channel_index",
        "reference_channel_index",
        "source_capture_commit_sha",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Stage-2 plant binding JSON key 집합이 exact하지 않습니다")
    contract = Stage2TwoKilohertzContract.canonical()
    expected_contract = {"id": contract.contract_id, "sha256": contract.digest()}
    exact = {
        "schema": STAGE2_2KHZ_PLANT_BINDING_SCHEMA,
        "status": "PASS",
        "canonical_training_eligible": True,
        "fixture_only": False,
        "control_band_contract": expected_contract,
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "verified_physical_subbands_hz": [
            list(row) for row in contract.physical_identification_subbands_hz
        ],
        "minimum_subband_consistency": 0.95,
        "maximum_timing_residual_samples": 0.270208,
        "minimum_independent_epochs_per_role": 8,
        "periodic_repeat_indices_allowed": False,
        "handoff_extra_samples": 256,
        "lead_source": "PlantDelays.lead()",
    }
    for key, expected in exact.items():
        if payload[key] != expected:
            raise ValueError(f"Stage-2 plant binding {key}가 canonical 계약과 다릅니다")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload["source_capture_commit_sha"])):
        raise ValueError("Stage-2 binding source capture commit이 40자리 SHA가 아닙니다")

    primary_path, primary_bytes, primary_sha = _artifact_ref(
        root, payload["primary_path"], label="Stage-2 primary path"
    )
    secondary_path, secondary_bytes, secondary_sha = _artifact_ref(
        root, payload["secondary_path"], label="Stage-2 secondary path"
    )
    raw_path, raw_bytes, raw_sha = _artifact_ref(root, payload["raw_capture"], label="Stage-2 raw")
    analysis_path, analysis_bytes, analysis_sha = _artifact_ref(
        root, payload["analysis"], label="Stage-2 analysis"
    )
    _, level_bytes, level_sha = _artifact_ref(
        root,
        payload["measurement_level_evidence"],
        label="Stage-2 measurement level",
    )
    _, clock_bytes, clock_sha = _artifact_ref(
        root,
        payload["relative_clock_model_receipt"],
        label="Stage-2 relative clock model",
    )
    try:
        level = json.loads(level_bytes.decode("utf-8"))
        clock = json.loads(clock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 level/relative clock receipt는 UTF-8 JSON이어야 합니다") from exc
    if not isinstance(level, dict) or not (
        level.get("schema") == "measurement_level_evidence_v2_bootstrap_pair"
        and level.get("passed") is True
        and int(level.get("sample_rate", 0)) == 48_000
        and float(level.get("probe_amplitude", -1.0)) == 0.003
        and level.get("same_amplifier_setting") is True
    ):
        raise ValueError("Stage-2 measurement-level evidence가 official PASS가 아닙니다")
    raw_analysis = _validate_raw_analysis_bytes(
        raw_bytes,
        analysis_bytes,
        raw_sha256=raw_sha,
    )
    expected_clock = {
        "schema": STAGE2_2KHZ_RELATIVE_CLOCK_MODEL_SCHEMA,
        "status": "PASS",
        "control_band_contract_sha256": contract.digest(),
        "raw_capture_sha256": raw_sha,
        "analysis_sha256": analysis_sha,
        "submitted_output_pcm_sha256": raw_analysis["submitted_output_pcm_sha256"],
        "relative_shared_q_model_pass": True,
        "absolute_hardware_frame_clock": False,
        "submitted_stereo_known_code_bound": True,
        "xrun_clip_status_slip_zero": True,
    }
    if not isinstance(clock, dict) or clock != expected_clock:
        raise ValueError(
            "Stage-2 relative shared-q clock model이 raw/analysis/submitted PCM에 exact 결속되지 않았습니다"
        )

    raw_relative = raw_path.relative_to(root).as_posix()
    analysis_relative = analysis_path.relative_to(root).as_posix()
    primary, primary_meta = _validate_path_npz(
        primary_bytes,
        role="primary",
        file_sha256=primary_sha,
        binding_sha256=binding_sha,
        raw_path=raw_relative,
        raw_sha256=raw_sha,
        analysis_path=analysis_relative,
        analysis_sha256=analysis_sha,
    )
    secondary, secondary_meta = _validate_path_npz(
        secondary_bytes,
        role="secondary",
        file_sha256=secondary_sha,
        binding_sha256=binding_sha,
        raw_path=raw_relative,
        raw_sha256=raw_sha,
        analysis_path=analysis_relative,
        analysis_sha256=analysis_sha,
    )
    if primary_meta["capture_id"] != secondary_meta["capture_id"]:
        raise ValueError("Stage-2 P/S가 same capture가 아닙니다")
    if primary_meta["capture_id"] != raw_analysis["capture_id"]:
        raise ValueError("Stage-2 P/S capture ID가 실제 raw/analysis와 다릅니다")
    if any(
        meta["source_capture_commit_sha"] != raw_analysis["source_capture_commit_sha"]
        for meta in (primary_meta, secondary_meta)
    ) or payload["source_capture_commit_sha"] != raw_analysis["source_capture_commit_sha"]:
        raise ValueError("Stage-2 P/S/raw/analysis/binding source capture commit이 다릅니다")
    if primary_meta["delay_samples"] != raw_analysis["primary_delay_samples"]:
        raise ValueError("Stage-2 primary delay가 analysis와 다릅니다")
    if secondary_meta["delay_samples"] != raw_analysis["secondary_delay_samples"]:
        raise ValueError("Stage-2 secondary delay가 analysis와 다릅니다")
    if primary_meta["band_consistency"] != raw_analysis["primary_band_consistency"]:
        raise ValueError("Stage-2 primary consistency가 analysis와 다릅니다")
    if secondary_meta["band_consistency"] != raw_analysis["secondary_band_consistency"]:
        raise ValueError("Stage-2 secondary consistency가 analysis와 다릅니다")
    for meta in (primary_meta, secondary_meta):
        if not math.isclose(
            float(meta["timing_residual_max_samples"]),
            float(raw_analysis["timing_residual_max_samples"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Stage-2 P/S timing residual이 analysis와 다릅니다")
    err = int(payload["err_channel_index"])
    reference = int(payload["reference_channel_index"])
    for meta in (primary_meta, secondary_meta):
        if meta["err_channel_index"] != err or meta["reference_channel_index"] != reference:
            raise ValueError("Stage-2 binding과 P/S NPZ 채널 선택이 다릅니다")
    delays = PlantDelays(
        primary_delay_samples=int(primary.coarse_delay_samples),
        secondary_delay_samples=int(secondary.coarse_delay_samples),
        handoff_samples=256,
        sample_rate=48_000,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=primary.post_onset_fir,
        plant_delays=delays,
    )
    return Stage2TwoKilohertzPlantBinding(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        primary_operator=primary,
        secondary_operator=secondary,
        primary_path_sha256=primary_sha,
        secondary_path_sha256=secondary_sha,
        raw_capture_sha256=raw_sha,
        analysis_sha256=analysis_sha,
        measurement_level_evidence_sha256=level_sha,
        relative_clock_model_receipt_sha256=clock_sha,
        verified_physical_subbands_hz=tuple(
            tuple(row) for row in contract.physical_identification_subbands_hz
        ),
        err_channel_index=err,
        reference_channel_index=reference,
        block_size=256,
        binding_file_sha256=binding_sha,
        source_capture_commit_sha=raw_analysis["source_capture_commit_sha"],
        fixture_only=True,
    )


def load_stage2_2khz_plant_binding(
    binding_path: str | Path,
    *,
    repository_root: str | Path,
    expected_binding_file_sha256: str | None = None,
) -> Stage2TwoKilohertzPlantBinding:
    """human-reviewed clean Git anchor를 먼저 검증하는 production P/S 경계."""

    root = Path(repository_root).resolve(strict=True)
    authority, _, head = verify_tracked_head_authority(
        root, STAGE2_2KHZ_PHYSICAL_AUTHORITY_PATH
    )
    contract = Stage2TwoKilohertzContract.canonical()
    expected_keys = {
        "schema",
        "authority_kind",
        "status",
        "source_capture_commit_sha",
        "control_band_contract_sha256",
        "binding_path",
        "binding_file_sha256",
        "primary_path_sha256",
        "secondary_path_sha256",
        "raw_capture_sha256",
        "analysis_sha256",
        "measurement_level_evidence_sha256",
        "relative_clock_model_receipt_sha256",
    }
    if not isinstance(authority, dict) or set(authority) != expected_keys:
        raise ValueError("Stage-2 physical Git authority key 집합이 exact하지 않습니다")
    if (
        authority["schema"] != STAGE2_2KHZ_PHYSICAL_AUTHORITY_SCHEMA
        or authority["authority_kind"] != "human_reviewed_physical_capture"
        or authority["status"] != "APPROVED"
        or authority["control_band_contract_sha256"] != contract.digest()
        or not re.fullmatch(r"[0-9a-f]{40}", str(authority["source_capture_commit_sha"]))
    ):
        raise ValueError("Stage-2 physical Git authority 의미가 canonical과 다릅니다")
    verify_source_commit_ancestor(
        root, str(authority["source_capture_commit_sha"]), head=head
    )
    candidate = Path(binding_path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Stage-2 binding은 repository 내부에 있어야 합니다") from exc
    else:
        relative = candidate
    if authority["binding_path"] != relative.as_posix():
        raise ValueError("Stage-2 physical authority binding path가 요청과 다릅니다")
    authority_binding_sha = _require_sha256(
        authority["binding_file_sha256"], label="physical authority binding SHA"
    )
    if expected_binding_file_sha256 is not None and authority_binding_sha != _require_sha256(
        expected_binding_file_sha256, label="expected Stage-2 binding SHA"
    ):
        raise ValueError("Stage-2 physical authority/profile binding SHA가 다릅니다")
    parsed = _load_self_attested_stage2_2khz_plant_binding_for_test(
        binding_path,
        repository_root=root,
        expected_binding_file_sha256=authority_binding_sha,
    )
    actual = {
        "primary_path_sha256": parsed.primary_path_sha256,
        "secondary_path_sha256": parsed.secondary_path_sha256,
        "raw_capture_sha256": parsed.raw_capture_sha256,
        "analysis_sha256": parsed.analysis_sha256,
        "measurement_level_evidence_sha256": parsed.measurement_level_evidence_sha256,
        "relative_clock_model_receipt_sha256": parsed.relative_clock_model_receipt_sha256,
    }
    if any(authority[key] != value for key, value in actual.items()):
        raise ValueError("Stage-2 physical authority가 실제 P/S/raw/analysis bytes와 다릅니다")
    if authority["source_capture_commit_sha"] != parsed.source_capture_commit_sha:
        raise ValueError("Stage-2 physical authority source capture commit이 raw와 다릅니다")
    return replace(parsed, fixture_only=False)


__all__ = [
    "STAGE2_2KHZ_ANALYSIS_SCHEMA",
    "STAGE2_2KHZ_PATH_NPZ_SCHEMA",
    "STAGE2_2KHZ_PLANT_BINDING_SCHEMA",
    "STAGE2_2KHZ_RAW_CAPTURE_SCHEMA",
    "STAGE2_2KHZ_RELATIVE_CLOCK_MODEL_SCHEMA",
    "STAGE2_2KHZ_PHYSICAL_AUTHORITY_PATH",
    "STAGE2_2KHZ_PHYSICAL_AUTHORITY_SCHEMA",
    "Stage2SourceOperatingLevelBinding",
    "Stage2TwoKilohertzPlantBinding",
    "load_stage2_2khz_plant_binding",
]
