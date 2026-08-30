"""현행 덕트의 digital/acoustic-reference 인과성 read-only 감사.

이 모듈은 오디오 장치를 열지 않는다. ``configs/duct.yaml``과 그 파일이 가리키는
strict P/S NPZ 및 immutable raw/analysis SHA를 읽어 snapshot을 만들고, 계산은 frozen
schema와 순수 함수로 수행한다.

중요한 구분은 다음과 같다.

* Jetson이 앞으로 재생할 WAV/자연음을 이미 갖고 있는 경우는 digital source playback이다.
* 외부 자연음을 upstream REF mic로 처음 관측하는 경우는 acoustic reference다.

둘 다 "처음 듣는 소리"일 수 있지만 전자는 미래 playback ``U_k``를 알고 있고 후자는
그렇지 않다. acoustic 경로의 미측정 latency 항을 기하 추정으로 채워 PASS하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .control_band_contract import BroadbandFullOctaveContractV3
from .timing import PlantDelays, TrainingTimingContract


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{name} 는 lowercase SHA-256이어야 합니다")


def phase_error_budget(
    *,
    frequency_hz: float,
    target_attenuation_db: float,
    sample_rate: int,
) -> tuple[float, float, float]:
    """동일 진폭 두 파형의 위상 오차만 있을 때 허용 budget을 계산한다.

    residual ratio는 ``2*sin(|phi|/2)``다. 반환값은
    ``(max_phase_radian, max_phase_degree, max_timing_error_samples)``다.
    amplitude mismatch, plant error, mode shape는 포함하지 않으므로 성능 약속이 아니다.
    """

    frequency = float(frequency_hz)
    attenuation = float(target_attenuation_db)
    fs = int(sample_rate)
    if not math.isfinite(frequency) or not 0.0 < frequency < fs / 2.0:
        raise ValueError("frequency_hz는 0과 Nyquist 사이여야 합니다")
    if not math.isfinite(attenuation) or attenuation <= 0.0:
        raise ValueError("target_attenuation_db는 양수여야 합니다")
    residual_ratio = 10.0 ** (-attenuation / 20.0)
    phase_radian = 2.0 * math.asin(residual_ratio / 2.0)
    samples = phase_radian * fs / (2.0 * math.pi * frequency)
    return phase_radian, math.degrees(phase_radian), samples


def first_transverse_mode_hz(
    *, width_m: float, height_m: float, speed_of_sound_mps: float
) -> float:
    """직사각 덕트의 가장 낮은 (1,0)/(0,1) 횡모드 cutoff."""

    width = float(width_m)
    height = float(height_m)
    speed = float(speed_of_sound_mps)
    if min(width, height, speed) <= 0.0 or not all(
        math.isfinite(value) for value in (width, height, speed)
    ):
        raise ValueError("덕트 치수와 음속은 유한한 양수여야 합니다")
    return speed / (2.0 * max(width, height))


def propagating_rectangular_mode_count(
    *,
    frequency_hz: float,
    width_m: float,
    height_m: float,
    speed_of_sound_mps: float,
) -> int:
    """비음수 (m,n) 직사각 waveguide mode 중 cutoff 이하 개수(평면파 포함)."""

    frequency = float(frequency_hz)
    width = float(width_m)
    height = float(height_m)
    speed = float(speed_of_sound_mps)
    if frequency <= 0.0 or min(width, height, speed) <= 0.0:
        raise ValueError("frequency/geometry는 양수여야 합니다")
    max_m = int(math.floor(2.0 * frequency * width / speed))
    max_n = int(math.floor(2.0 * frequency * height / speed))
    count = 0
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            cutoff = speed * 0.5 * math.sqrt((m / width) ** 2 + (n / height) ** 2)
            if cutoff <= frequency + 1.0e-12:
                count += 1
    return count


class StrictPathMetadata(BaseModel):
    model_config = _FROZEN

    role: Literal["primary", "secondary"]
    relative_path: str
    artifact_sha256: str
    sample_rate: int
    calibration_block_size: int
    delay_samples: int
    bulk_delay_samples: int
    delay_semantics: str
    fir_taps: int
    fir_peak_offset_samples: int
    capture_id: str
    source_raw_npz_path: str
    source_raw_npz_sha256: str
    source_analysis_npz_path: str
    source_analysis_npz_sha256: str
    excitation_band_hz: tuple[float, float]
    consistency_band_hz: tuple[float, float]
    band_consistency_hz: tuple[tuple[float, float], ...]
    band_consistency: tuple[float, ...]
    kept_repeat_indices: tuple[int, ...]
    anchor_repeat: int
    xrun_count: int
    output_channel: Literal["noise", "cancel"]
    output_pcm_provenance: Literal["observed_submitted_int16"]

    @model_validator(mode="after")
    def _validate_path(self) -> "StrictPathMetadata":
        for name in (
            "artifact_sha256",
            "source_raw_npz_sha256",
            "source_analysis_npz_sha256",
        ):
            _require_sha256(name, str(getattr(self, name)))
        if self.sample_rate <= 0 or self.calibration_block_size <= 0:
            raise ValueError("strict path sample rate/block은 양수여야 합니다")
        if min(self.delay_samples, self.bulk_delay_samples, self.fir_peak_offset_samples) < 0:
            raise ValueError("strict path delay는 0 이상이어야 합니다")
        if self.fir_taps <= 0 or not self.kept_repeat_indices:
            raise ValueError("strict path FIR/kept repeats가 비어 있습니다")
        if self.xrun_count != 0:
            raise ValueError("strict path xrun은 0이어야 합니다")
        if len(self.band_consistency_hz) != len(self.band_consistency):
            raise ValueError("strict path band consistency 길이가 다릅니다")
        if self.role == "primary" and self.output_channel != "noise":
            raise ValueError("primary path output_channel은 noise여야 합니다")
        if self.role == "secondary" and self.output_channel != "cancel":
            raise ValueError("secondary path output_channel은 cancel이어야 합니다")
        return self


class DuctGeometrySnapshot(BaseModel):
    model_config = _FROZEN

    config_relative_path: str
    config_sha256: str
    interior_length_m: float
    cross_section_m: tuple[float, float]
    speed_of_sound_mps: float
    noise_speaker_x_m: float
    reference_mic_x_m: float
    cancel_speaker_x_m: float
    error_mic_x_m: float
    configured_plane_wave_cutoff_hz: float
    computed_first_transverse_mode_hz: float
    geometry_authority: Literal["repository_config_only_not_field_verified"] = (
        "repository_config_only_not_field_verified"
    )
    error_mic_position_authority: Literal[
        "repository_comment_provisional", "repository_config_unmarked"
    ]

    @model_validator(mode="after")
    def _validate_geometry(self) -> "DuctGeometrySnapshot":
        _require_sha256("config_sha256", self.config_sha256)
        if self.interior_length_m <= 0.0 or min(self.cross_section_m) <= 0.0:
            raise ValueError("덕트 내부 치수는 양수여야 합니다")
        expected = first_transverse_mode_hz(
            width_m=self.cross_section_m[0],
            height_m=self.cross_section_m[1],
            speed_of_sound_mps=self.speed_of_sound_mps,
        )
        if not math.isclose(
            self.computed_first_transverse_mode_hz,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("computed transverse cutoff가 geometry와 다릅니다")
        # YAML의 1633은 정수 반올림 문서값이다. 큰 차이는 fail-closed한다.
        if abs(self.configured_plane_wave_cutoff_hz - expected) > 1.0:
            raise ValueError("configured plane-wave cutoff가 geometry 계산과 다릅니다")
        return self

    @property
    def geometric_ref_to_err_advance_samples(self) -> float:
        distance = self.error_mic_x_m - self.reference_mic_x_m
        return distance / self.speed_of_sound_mps * 48_000.0


class PlantTimingSnapshot(BaseModel):
    model_config = _FROZEN

    primary: StrictPathMetadata
    secondary: StrictPathMetadata
    plant_delays: PlantDelays
    training_timing_contract: TrainingTimingContract
    training_timing_contract_sha256: str
    derived_lead_samples: int
    derived_raw_lead_samples: int
    strict_capture_hardware_input_card: str
    strict_capture_hardware_output_product: str
    strict_raw_file_sha_verified: Literal[True] = True
    strict_analysis_file_sha_verified: Literal[True] = True

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "PlantTimingSnapshot":
        if self.primary.capture_id != self.secondary.capture_id:
            raise ValueError("P/S capture_id가 다릅니다")
        if (
            self.primary.source_raw_npz_sha256
            != self.secondary.source_raw_npz_sha256
            or self.primary.source_analysis_npz_sha256
            != self.secondary.source_analysis_npz_sha256
        ):
            raise ValueError("P/S raw/analysis SHA가 다릅니다")
        if (
            self.primary.sample_rate != self.secondary.sample_rate
            or self.primary.calibration_block_size
            != self.secondary.calibration_block_size
            or self.primary.anchor_repeat != self.secondary.anchor_repeat
            or self.primary.kept_repeat_indices != self.secondary.kept_repeat_indices
        ):
            raise ValueError("P/S sample/block/anchor/kept repeat가 다릅니다")
        lead = self.plant_delays.lead()
        if self.derived_lead_samples != int(lead.samples):
            raise ValueError("lead는 PlantDelays.lead() 결과여야 합니다")
        if self.derived_raw_lead_samples != int(lead.raw_samples):
            raise ValueError("raw lead는 PlantDelays.lead() 결과여야 합니다")
        if self.training_timing_contract_sha256 != self.training_timing_contract.digest():
            raise ValueError("TrainingTimingContract digest가 다릅니다")
        if (
            self.training_timing_contract.digital_reference_lead_samples
            != self.derived_lead_samples
        ):
            raise ValueError("TrainingTimingContract와 PlantDelays lead가 다릅니다")
        return self


class OctavePhaseBudget(BaseModel):
    model_config = _FROZEN

    center_hz: float
    lower_hz: float
    upper_hz: float
    target_attenuation_db: float
    maximum_phase_error_degree: float
    maximum_timing_error_samples_at_center: float
    maximum_timing_error_samples_at_upper_edge: float
    phase_only_equal_amplitude_assumption: Literal[True] = True
    physical_performance_claim: Literal[False] = False
    modal_regime: Literal[
        "plane_wave_band", "crosses_first_transverse_cutoff", "higher_order_band"
    ]
    propagating_mode_count_at_center: int
    minimum_spatial_err_positions_for_quiet_zone: Literal[5] = 5
    single_point_is_quiet_zone_evidence: Literal[False] = False


class DigitalReferenceAssessment(BaseModel):
    model_config = _FROZEN

    causality_status: Literal["CONDITIONALLY_CAUSAL"] = "CONDITIONALLY_CAUSAL"
    source_requirement: Literal["jetson_generated_future_playback_samples"] = (
        "jetson_generated_future_playback_samples"
    )
    derived_lead_samples: int
    handoff_samples: int
    primary_delay_samples: int
    secondary_delay_samples: int
    timing_authority: Literal["PlantDelays_and_TrainingTimingContract"] = (
        "PlantDelays_and_TrainingTimingContract"
    )
    strict_plant_trusted_band_hz: tuple[float, float]
    broadband_125_to_8000_octave_status: Literal["BLOCKED"] = "BLOCKED"
    broadband_blockers: tuple[str, ...]
    physical_attenuation_pass: Literal[False] = False


class AcousticReferenceRuntimeEvidence(BaseModel):
    """같은 runtime/timeline에서 직접 측정된 경우에만 값을 채우는 latency 증거."""

    model_config = _FROZEN

    schema_version: Literal["acoustic_reference_runtime_latency_evidence_v1"] = (
        "acoustic_reference_runtime_latency_evidence_v1"
    )
    ref_to_err_advance_samples: float | None = None
    ref_to_err_advance_receipt_sha256: str | None = None
    adc_observation_latency_samples: float | None = None
    adc_observation_receipt_sha256: str | None = None
    inference_p99_samples: float | None = None
    inference_receipt_sha256: str | None = None
    dac_output_latency_samples: float | None = None
    dac_output_receipt_sha256: str | None = None
    secondary_acoustic_delay_samples: float | None = None
    secondary_acoustic_receipt_sha256: str | None = None
    common_runtime_timeline_receipt_sha256: str | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> "AcousticReferenceRuntimeEvidence":
        pairs = (
            ("ref_to_err_advance_samples", "ref_to_err_advance_receipt_sha256"),
            ("adc_observation_latency_samples", "adc_observation_receipt_sha256"),
            ("inference_p99_samples", "inference_receipt_sha256"),
            ("dac_output_latency_samples", "dac_output_receipt_sha256"),
            ("secondary_acoustic_delay_samples", "secondary_acoustic_receipt_sha256"),
        )
        for value_name, sha_name in pairs:
            value = getattr(self, value_name)
            sha = getattr(self, sha_name)
            if (value is None) != (sha is None):
                raise ValueError(f"{value_name}와 {sha_name}는 함께 있어야 합니다")
            if value is not None:
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"{value_name}는 유한한 0 이상이어야 합니다")
                _require_sha256(sha_name, str(sha))
        if self.common_runtime_timeline_receipt_sha256 is not None:
            _require_sha256(
                "common_runtime_timeline_receipt_sha256",
                self.common_runtime_timeline_receipt_sha256,
            )
        return self


class AcousticReferenceAssessment(BaseModel):
    model_config = _FROZEN

    causality_status: Literal["CONDITIONALLY_CAUSAL", "BLOCKED"]
    broadband_random_live_sound_status: Literal["CONDITIONALLY_CAUSAL", "BLOCKED"]
    periodic_predictive_sound_status: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"
    missing_or_unbound_terms: tuple[str, ...]
    ref_to_err_advance_samples: float | None
    required_latency_samples: float | None
    causal_margin_samples: float | None
    fixed_handoff_samples: int
    inference_deadline_met: bool | None
    strict_secondary_calibration_delay_samples: int
    strict_secondary_is_decomposed_runtime_latency: Literal[False] = False
    geometric_ref_to_err_estimate_samples: float
    geometric_estimate_is_canonical_measurement: Literal[False] = False
    broadband_125_to_8000_octave_status: Literal["BLOCKED"] = "BLOCKED"
    physical_attenuation_pass: Literal[False] = False


class NaturalSoundRouteAssessment(BaseModel):
    model_config = _FROZEN

    origin: Literal[
        "new_file_or_recording_replayed_by_jetson",
        "live_sound_first_observed_by_upstream_ref_mic",
    ]
    reference_mode: Literal["digital_reference", "acoustic_reference"]
    causality_status: Literal["CONDITIONALLY_CAUSAL", "BLOCKED"]
    statement: str


class CurrentReferenceModeAudit(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["current_reference_mode_causality_audit_v1"] = (
        "current_reference_mode_causality_audit_v1"
    )
    authority: Literal["read_only_structural_audit_not_anc_performance"] = (
        "read_only_structural_audit_not_anc_performance"
    )
    geometry: DuctGeometrySnapshot
    timing: PlantTimingSnapshot
    phase_budgets_10db: tuple[OctavePhaseBudget, ...]
    phase_budgets_20db: tuple[OctavePhaseBudget, ...]
    digital_reference: DigitalReferenceAssessment
    acoustic_reference: AcousticReferenceAssessment
    natural_sound_routes: tuple[NaturalSoundRouteAssessment, ...]
    overall_broadband_deployment_status: Literal["BLOCKED"] = "BLOCKED"
    physical_attenuation_pass: Literal[False] = False

    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


def _required_npz_scalar(data: Any, key: str) -> Any:
    if key not in data.files:
        raise ValueError(f"strict path NPZ에 {key}가 없습니다")
    array = np.asarray(data[key])
    if array.shape != ():
        raise ValueError(f"strict path {key}는 scalar여야 합니다")
    return array.item()


def _load_strict_path(
    *, repo_root: Path, relative_path: str, role: Literal["primary", "secondary"]
) -> tuple[StrictPathMetadata, np.ndarray]:
    path = (repo_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        fir = np.asarray(data["fir"])
        metadata = StrictPathMetadata(
            role=role,
            relative_path=relative_path,
            artifact_sha256=_sha256_file(path),
            sample_rate=int(_required_npz_scalar(data, "sample_rate")),
            calibration_block_size=int(
                _required_npz_scalar(data, "calibration_block_size")
            ),
            delay_samples=int(_required_npz_scalar(data, "delay_samples")),
            bulk_delay_samples=int(_required_npz_scalar(data, "bulk_delay_samples")),
            delay_semantics=str(_required_npz_scalar(data, "delay_semantics")),
            fir_taps=int(fir.size),
            fir_peak_offset_samples=int(np.argmax(np.abs(fir))),
            capture_id=str(_required_npz_scalar(data, "capture_id")),
            source_raw_npz_path=str(
                _required_npz_scalar(data, "source_raw_npz_path")
            ),
            source_raw_npz_sha256=str(
                _required_npz_scalar(data, "source_raw_npz_sha256")
            ),
            source_analysis_npz_path=str(
                _required_npz_scalar(data, "source_analysis_npz_path")
            ),
            source_analysis_npz_sha256=str(
                _required_npz_scalar(data, "source_analysis_npz_sha256")
            ),
            excitation_band_hz=tuple(
                float(value) for value in np.asarray(data["excitation_band_hz"])
            ),
            consistency_band_hz=tuple(
                float(value) for value in np.asarray(data["consistency_band_hz"])
            ),
            band_consistency_hz=tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(data["band_consistency_hz"])
            ),
            band_consistency=tuple(
                float(value) for value in np.asarray(data["band_consistency"])
            ),
            kept_repeat_indices=tuple(
                int(value) for value in np.asarray(data["kept_repeat_indices"])
            ),
            anchor_repeat=int(_required_npz_scalar(data, "anchor_repeat")),
            xrun_count=int(_required_npz_scalar(data, "xrun_count")),
            output_channel=str(_required_npz_scalar(data, "output_channel")),
            output_pcm_provenance=str(
                _required_npz_scalar(data, "output_pcm_provenance")
            ),
        )
    return metadata, np.ascontiguousarray(fir, dtype=np.float32)


def load_current_causality_snapshot(repo_root: str | Path) -> tuple[
    DuctGeometrySnapshot, PlantTimingSnapshot
]:
    """현 checkout의 duct/strict P/S/raw/analysis를 read-only로 검증해 읽는다."""

    root = Path(repo_root).resolve()
    config_path = root / "configs/duct.yaml"
    config_bytes = config_path.read_bytes()
    config_text = config_bytes.decode("utf-8")
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError("configs/duct.yaml이 mapping이 아닙니다")

    duct = config["duct"]
    positions = config["positions_m"]
    acoustics = config["acoustics"]
    cross = tuple(float(value) for value in duct["cross_section_m"])
    if len(cross) != 2:
        raise ValueError("duct.cross_section_m은 2개 값이어야 합니다")
    cutoff = first_transverse_mode_hz(
        width_m=cross[0],
        height_m=cross[1],
        speed_of_sound_mps=float(duct["speed_of_sound_mps"]),
    )
    error_line = next(
        (line for line in config_text.splitlines() if line.lstrip().startswith("error_mic:")),
        "",
    )
    geometry = DuctGeometrySnapshot(
        config_relative_path="configs/duct.yaml",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        interior_length_m=float(duct["interior_length_m"]),
        cross_section_m=cross,
        speed_of_sound_mps=float(duct["speed_of_sound_mps"]),
        noise_speaker_x_m=float(positions["noise_speaker"]),
        reference_mic_x_m=float(positions["reference_mic"]),
        cancel_speaker_x_m=float(positions["cancel_speaker"]),
        error_mic_x_m=float(positions["error_mic"]),
        configured_plane_wave_cutoff_hz=float(acoustics["plane_wave_cutoff_hz"]),
        computed_first_transverse_mode_hz=cutoff,
        error_mic_position_authority=(
            "repository_comment_provisional"
            if "잠정" in error_line
            else "repository_config_unmarked"
        ),
    )

    primary_rel = str(config["digital_reference"]["primary_path_npz"])
    secondary_rel = str(config["secondary_path"]["npz"])
    primary, primary_fir = _load_strict_path(
        repo_root=root, relative_path=primary_rel, role="primary"
    )
    secondary, _ = _load_strict_path(
        repo_root=root, relative_path=secondary_rel, role="secondary"
    )

    raw_path = root / primary.source_raw_npz_path
    analysis_path = root / primary.source_analysis_npz_path
    if _sha256_file(raw_path) != primary.source_raw_npz_sha256:
        raise ValueError("strict raw file SHA가 P/S metadata와 다릅니다")
    if _sha256_file(analysis_path) != primary.source_analysis_npz_sha256:
        raise ValueError("strict analysis file SHA가 P/S metadata와 다릅니다")
    with np.load(raw_path, allow_pickle=False) as raw:
        raw_metadata = json.loads(str(_required_npz_scalar(raw, "metadata_json")))
    hardware = raw_metadata["hardware_identity"]

    plant_delays = PlantDelays.from_config(
        duct_cfg=config,
        secondary_delay_samples=secondary.delay_samples,
        primary_delay_samples=primary.delay_samples,
        sample_rate=primary.sample_rate,
    )
    timing_contract = TrainingTimingContract.derive(
        primary_fir=primary_fir,
        plant_delays=plant_delays,
    )
    lead = plant_delays.lead()
    timing = PlantTimingSnapshot(
        primary=primary,
        secondary=secondary,
        plant_delays=plant_delays,
        training_timing_contract=timing_contract,
        training_timing_contract_sha256=timing_contract.digest(),
        derived_lead_samples=int(lead.samples),
        derived_raw_lead_samples=int(lead.raw_samples),
        strict_capture_hardware_input_card=str(hardware["input"]["card"]),
        strict_capture_hardware_output_product=str(
            hardware["physical_fingerprint"]["output"]["stable_attributes"][0][
                "values"
            ]["product"]
        ),
    )
    return geometry, timing


def build_octave_phase_budgets(
    *,
    geometry: DuctGeometrySnapshot,
    target_attenuation_db: float,
    sample_rate: int,
) -> tuple[OctavePhaseBudget, ...]:
    """canonical v3 7개 exact octave의 phase-only timing budget."""

    contract = BroadbandFullOctaveContractV3.canonical()
    rows: list[OctavePhaseBudget] = []
    cutoff = geometry.computed_first_transverse_mode_hz
    for center, (lower, upper) in zip(
        contract.octave_objective_centers_hz,
        contract.equal_weight_octave_objective_bands_hz,
        strict=True,
    ):
        _, phase_deg, center_samples = phase_error_budget(
            frequency_hz=center,
            target_attenuation_db=target_attenuation_db,
            sample_rate=sample_rate,
        )
        _, _, upper_samples = phase_error_budget(
            frequency_hz=upper,
            target_attenuation_db=target_attenuation_db,
            sample_rate=sample_rate,
        )
        if upper <= cutoff:
            regime = "plane_wave_band"
        elif lower < cutoff < upper:
            regime = "crosses_first_transverse_cutoff"
        else:
            regime = "higher_order_band"
        rows.append(
            OctavePhaseBudget(
                center_hz=center,
                lower_hz=lower,
                upper_hz=upper,
                target_attenuation_db=target_attenuation_db,
                maximum_phase_error_degree=phase_deg,
                maximum_timing_error_samples_at_center=center_samples,
                maximum_timing_error_samples_at_upper_edge=upper_samples,
                modal_regime=regime,
                propagating_mode_count_at_center=propagating_rectangular_mode_count(
                    frequency_hz=center,
                    width_m=geometry.cross_section_m[0],
                    height_m=geometry.cross_section_m[1],
                    speed_of_sound_mps=geometry.speed_of_sound_mps,
                ),
            )
        )
    return tuple(rows)


def assess_acoustic_reference(
    *,
    geometry: DuctGeometrySnapshot,
    timing: PlantTimingSnapshot,
    evidence: AcousticReferenceRuntimeEvidence,
) -> AcousticReferenceAssessment:
    """측정된 latency 항이 모두 같은 runtime timeline에 있을 때만 margin을 계산한다."""

    required_fields = (
        "ref_to_err_advance_samples",
        "adc_observation_latency_samples",
        "inference_p99_samples",
        "dac_output_latency_samples",
        "secondary_acoustic_delay_samples",
        "common_runtime_timeline_receipt_sha256",
    )
    missing = tuple(name for name in required_fields if getattr(evidence, name) is None)
    if missing:
        return AcousticReferenceAssessment(
            causality_status="BLOCKED",
            broadband_random_live_sound_status="BLOCKED",
            missing_or_unbound_terms=missing,
            ref_to_err_advance_samples=evidence.ref_to_err_advance_samples,
            required_latency_samples=None,
            causal_margin_samples=None,
            fixed_handoff_samples=timing.plant_delays.handoff_samples,
            inference_deadline_met=None,
            strict_secondary_calibration_delay_samples=timing.secondary.delay_samples,
            geometric_ref_to_err_estimate_samples=(
                geometry.geometric_ref_to_err_advance_samples
            ),
        )

    assert evidence.ref_to_err_advance_samples is not None
    assert evidence.adc_observation_latency_samples is not None
    assert evidence.inference_p99_samples is not None
    assert evidence.dac_output_latency_samples is not None
    assert evidence.secondary_acoustic_delay_samples is not None
    handoff = float(timing.plant_delays.handoff_samples)
    # inference compute는 이 one-block handoff 안에 완료돼야 하므로 다시 더하지 않는다.
    inference_deadline_met = evidence.inference_p99_samples <= handoff
    required = (
        evidence.adc_observation_latency_samples
        + handoff
        + evidence.dac_output_latency_samples
        + evidence.secondary_acoustic_delay_samples
    )
    margin = evidence.ref_to_err_advance_samples - required
    causal = bool(inference_deadline_met and margin >= 0.0)
    return AcousticReferenceAssessment(
        causality_status="CONDITIONALLY_CAUSAL" if causal else "BLOCKED",
        broadband_random_live_sound_status=(
            "CONDITIONALLY_CAUSAL" if causal else "BLOCKED"
        ),
        missing_or_unbound_terms=(),
        ref_to_err_advance_samples=evidence.ref_to_err_advance_samples,
        required_latency_samples=required,
        causal_margin_samples=margin,
        fixed_handoff_samples=timing.plant_delays.handoff_samples,
        inference_deadline_met=inference_deadline_met,
        strict_secondary_calibration_delay_samples=timing.secondary.delay_samples,
        geometric_ref_to_err_estimate_samples=(
            geometry.geometric_ref_to_err_advance_samples
        ),
    )


def build_current_reference_mode_audit(
    *,
    repo_root: str | Path,
    acoustic_evidence: AcousticReferenceRuntimeEvidence | None = None,
) -> CurrentReferenceModeAudit:
    geometry, timing = load_current_causality_snapshot(repo_root)
    acoustic = assess_acoustic_reference(
        geometry=geometry,
        timing=timing,
        evidence=acoustic_evidence or AcousticReferenceRuntimeEvidence(),
    )
    trusted_lower = max(
        timing.primary.consistency_band_hz[0],
        timing.secondary.consistency_band_hz[0],
    )
    trusted_upper = min(
        timing.primary.consistency_band_hz[1],
        timing.secondary.consistency_band_hz[1],
    )
    digital = DigitalReferenceAssessment(
        derived_lead_samples=timing.derived_lead_samples,
        handoff_samples=timing.plant_delays.handoff_samples,
        primary_delay_samples=timing.plant_delays.primary_delay_samples,
        secondary_delay_samples=timing.plant_delays.secondary_delay_samples,
        strict_plant_trusted_band_hz=(trusted_lower, trusted_upper),
        broadband_blockers=(
            "strict P/S consistency authority ends at 1600 Hz",
            "125 Hz octave lower edge 88.388 Hz is outside strict consistency authority",
            "2/4/8 kHz multi-panel P/S and five-position physical quiet-zone are absent",
            "output-clock-master physical clock witness and canonical ref-only model are absent",
        ),
    )
    routes = (
        NaturalSoundRouteAssessment(
            origin="new_file_or_recording_replayed_by_jetson",
            reference_mode="digital_reference",
            causality_status=digital.causality_status,
            statement=(
                "처음 듣는 speech/music/environment/machine이어도 Jetson이 U_k를 먼저 "
                "읽고 재생하면 digital-reference 조건부 인과 경로다"
            ),
        ),
        NaturalSoundRouteAssessment(
            origin="live_sound_first_observed_by_upstream_ref_mic",
            reference_mode="acoustic_reference",
            causality_status=acoustic.causality_status,
            statement=(
                "실시간 외부 자연음은 미래 U_k가 없으므로 측정된 REF→ERR advance와 "
                "ADC/handoff/DAC/S latency margin 없이는 차단된다"
            ),
        ),
    )
    return CurrentReferenceModeAudit(
        geometry=geometry,
        timing=timing,
        phase_budgets_10db=build_octave_phase_budgets(
            geometry=geometry,
            target_attenuation_db=10.0,
            sample_rate=timing.plant_delays.sample_rate,
        ),
        phase_budgets_20db=build_octave_phase_budgets(
            geometry=geometry,
            target_attenuation_db=20.0,
            sample_rate=timing.plant_delays.sample_rate,
        ),
        digital_reference=digital,
        acoustic_reference=acoustic,
        natural_sound_routes=routes,
    )


__all__ = [
    "AcousticReferenceAssessment",
    "AcousticReferenceRuntimeEvidence",
    "CurrentReferenceModeAudit",
    "DigitalReferenceAssessment",
    "DuctGeometrySnapshot",
    "NaturalSoundRouteAssessment",
    "OctavePhaseBudget",
    "PlantTimingSnapshot",
    "StrictPathMetadata",
    "assess_acoustic_reference",
    "build_current_reference_mode_audit",
    "build_octave_phase_budgets",
    "first_transverse_mode_hz",
    "load_current_causality_snapshot",
    "phase_error_budget",
    "propagating_rectangular_mode_count",
]
