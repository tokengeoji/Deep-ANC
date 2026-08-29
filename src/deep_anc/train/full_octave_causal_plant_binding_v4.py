"""Full-octave causal ``S*y`` prefix consumer의 아직-비공개 binding boundary.

현재 Jetson에는 raw-bound 125 Hz--8 kHz P/S authority가 없다. 그러므로 이 module은
일반 descriptor를 public constructor로 받아 ``future fullband``라고 재표기하는 길을
제공하지 않는다. 실제 raw/analysis/electrical-witness/operator bytes를 검증하는 future
publisher가 생기기 전에는 test-fixture issuer만 존재하며, public adapter도 그 fixture를
거부한다.

이 module은 파일·NPZ·오디오 장치·GPU를 열지 않는다. SHA 문자열은 provenance의
식별자일 뿐 physical evidence를 스스로 발행하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from ..dsp.timing import TrainingTimingContract
from ..losses.broadband_loss import CausalFIRPathData


FULL_OCTAVE_CAUSAL_PLANT_BINDING_SCHEMA_V4 = (
    "full_octave_causal_plant_binding_v4"
)
FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4 = (
    "full_octave_causal_plant_authority_v4"
)
FULL_OCTAVE_CAUSAL_PREFIX_TRAINING_BLOCKER = (
    "BLOCKED_REQUIRES_SEPARATE_RAW_WITNESS_POPULATION_BATCH_DNH_AND_TRAINER_AUTHORITY"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _assert_operator_bytes(path: CausalFIRPathData, *, label: str) -> None:
    actual = hashlib.sha256(path.post_onset_fir.tobytes(order="C")).hexdigest()
    if actual != path.fir_sha256:
        raise ValueError(f"{label} FIR bytes가 declared SHA와 다릅니다")
    if path.post_onset_fir.flags.writeable:
        raise ValueError(f"{label} FIR은 immutable snapshot이어야 합니다")


def _frozen_operator_snapshot(path: CausalFIRPathData) -> CausalFIRPathData:
    """caller가 가진 writable ndarray와 수명/bytes를 공유하지 않는 1-D FIR snapshot."""

    values = np.asarray(path.post_onset_fir, dtype=np.dtype("<f8")).reshape(-1)
    # ``bytes`` backing은 writeable flag를 다시 올릴 수 없는 read-only buffer다.
    frozen_values = np.frombuffer(values.tobytes(order="C"), dtype=np.dtype("<f8"))
    snapshot = CausalFIRPathData(
        role=path.role,
        post_onset_fir=frozen_values,
        coarse_delay_samples=int(path.coarse_delay_samples),
        fractional_delay_samples=float(path.fractional_delay_samples),
        support_samples=int(path.support_samples),
        sample_rate=int(path.sample_rate),
        handoff_extra_samples=int(path.handoff_extra_samples),
        operator_file_sha256=path.operator_file_sha256,
        operator_internal_sha256=path.operator_internal_sha256,
        fir_sha256=path.fir_sha256,
        authority_sha256=path.authority_sha256,
        source_path=path.source_path,
        fractional_delay_encoded_in_post_onset_fir=bool(
            path.fractional_delay_encoded_in_post_onset_fir
        ),
    )
    snapshot.post_onset_fir.setflags(write=False)
    _assert_operator_bytes(snapshot, label=f"{path.role} operator")
    return snapshot


def _operator_payload(path: CausalFIRPathData) -> dict[str, object]:
    """FIR bytes SHA에 결속된 immutable operator descriptor."""

    _assert_operator_bytes(path, label=f"{path.role} operator")
    return {
        "role": path.role,
        "coarse_delay_samples": int(path.coarse_delay_samples),
        "fractional_delay_samples": float(path.fractional_delay_samples),
        "support_samples": int(path.support_samples),
        "sample_rate": int(path.sample_rate),
        "handoff_extra_samples": int(path.handoff_extra_samples),
        "operator_file_sha256": path.operator_file_sha256,
        "operator_internal_sha256": path.operator_internal_sha256,
        "fir_sha256": path.fir_sha256,
        "authority_sha256": path.authority_sha256,
        "source_path": path.source_path,
        "fractional_delay_encoded_in_post_onset_fir": bool(
            path.fractional_delay_encoded_in_post_onset_fir
        ),
    }


@dataclass(frozen=True, init=False)
class FullOctaveCausalPlantBindingV4:
    """future raw-bound issuer만 만들 수 있는 in-memory operator binding.

    현재는 `_for_test_fixture`만 존재한다. 이 fixture는 production adapter에서 거부된다.
    미래 publisher는 raw/analysis/witness/operator bytes를 실제로 재검증한 뒤 별도 private
    issuer를 추가해야 하며, 이 dataclass의 public construction을 열어서는 안 된다.
    """

    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    training_timing_contract: TrainingTimingContract
    training_timing_contract_sha256: str
    primary_operator: CausalFIRPathData
    secondary_operator: CausalFIRPathData
    verified_physical_subbands_hz: tuple[tuple[float, float], ...]
    raw_capture_sha256: str
    analysis_sha256: str
    primary_raw_capture_sha256: str
    secondary_raw_capture_sha256: str
    primary_analysis_sha256: str
    secondary_analysis_sha256: str
    plant_authority_sha256: str
    electrical_witness_receipt_sha256: str
    err_channel_index: int
    err_channel_selection_sha256: str
    reference_channel_index: int
    reference_channel_selection_sha256: str
    authority_schema: str
    block_size: int
    schema_version: str
    _fixture_only: bool

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "FullOctaveCausalPlantBindingV4는 raw-bound publisher만 발행할 수 있습니다; "
            "현재 production issuer는 아직 구현되지 않았습니다"
        )

    @classmethod
    def _for_test_fixture(cls, **values: Any) -> "FullOctaveCausalPlantBindingV4":
        """CPU regression 전용 issuer. public training path에서 절대 받아들이지 않는다."""

        return cls._issue(fixture_only=True, **values)

    @classmethod
    def _issue(cls, *, fixture_only: bool, **values: Any) -> "FullOctaveCausalPlantBindingV4":
        defaults: dict[str, Any] = {
            "authority_schema": FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4,
            "block_size": 256,
            "schema_version": FULL_OCTAVE_CAUSAL_PLANT_BINDING_SCHEMA_V4,
        }
        fields = (
            "control_band_contract",
            "control_band_contract_sha256",
            "training_timing_contract",
            "training_timing_contract_sha256",
            "primary_operator",
            "secondary_operator",
            "verified_physical_subbands_hz",
            "raw_capture_sha256",
            "analysis_sha256",
            "primary_raw_capture_sha256",
            "secondary_raw_capture_sha256",
            "primary_analysis_sha256",
            "secondary_analysis_sha256",
            "plant_authority_sha256",
            "electrical_witness_receipt_sha256",
            "err_channel_index",
            "err_channel_selection_sha256",
            "reference_channel_index",
            "reference_channel_selection_sha256",
            "authority_schema",
            "block_size",
            "schema_version",
        )
        supplied = {**defaults, **values}
        unexpected = set(supplied) - set(fields)
        missing = set(fields) - set(supplied)
        if unexpected or missing:
            raise ValueError(
                "causal plant binding issuer field가 정확하지 않습니다: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        instance = object.__new__(cls)
        for name in fields:
            object.__setattr__(instance, name, supplied[name])
        object.__setattr__(instance, "_fixture_only", bool(fixture_only))
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if not self._fixture_only:
            raise ValueError("raw-bound production issuer가 아직 없으므로 non-fixture binding은 금지됩니다")
        # Caller가 준 numpy storage와 binding이 공유되지 않도록 먼저 immutable bytes로 복제한다.
        object.__setattr__(self, "primary_operator", _frozen_operator_snapshot(self.primary_operator))
        object.__setattr__(self, "secondary_operator", _frozen_operator_snapshot(self.secondary_operator))
        if self.schema_version != FULL_OCTAVE_CAUSAL_PLANT_BINDING_SCHEMA_V4:
            raise ValueError("causal plant binding schema가 다릅니다")
        if self.authority_schema != FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4:
            raise ValueError(
                "full-octave causal plant authority v4가 아니므로 legacy/static evidence를 "
                "prefix adapter에 연결할 수 없습니다"
            )
        canonical = BroadbandFullOctaveContractV3.canonical()
        if self.control_band_contract != canonical:
            raise ValueError("canonical v3 full-octave control-band 계약이 필요합니다")
        if self.control_band_contract_sha256 != canonical.digest():
            raise ValueError("full-octave control-band contract SHA가 다릅니다")
        timing = self.training_timing_contract
        if int(timing.schema_version) != 2 or int(timing.sample_rate) != 48_000:
            raise ValueError("TrainingTimingContract v2/48 kHz가 필요합니다")
        if self.training_timing_contract_sha256 != timing.digest():
            raise ValueError("TrainingTimingContract payload/SHA가 다릅니다")
        if int(self.block_size) != 256:
            raise ValueError("canonical full-octave causal prefix block은 256 samples여야 합니다")
        if int(self.primary_operator.sample_rate) != int(timing.sample_rate):
            raise ValueError("primary operator와 timing sample rate가 다릅니다")
        if int(self.secondary_operator.sample_rate) != int(timing.sample_rate):
            raise ValueError("secondary operator와 timing sample rate가 다릅니다")
        if self.primary_operator.role != "primary":
            raise ValueError("primary operator role이 다릅니다")
        if self.secondary_operator.role != "secondary":
            raise ValueError("secondary operator role이 다릅니다")
        if int(self.primary_operator.handoff_extra_samples) != 0:
            raise ValueError("primary causal operator에는 handoff가 없어야 합니다")
        if int(self.secondary_operator.handoff_extra_samples) != int(
            timing.handoff_samples
        ):
            raise ValueError("secondary handoff와 TrainingTimingContract가 다릅니다")
        if int(self.primary_operator.coarse_delay_samples) != int(
            timing.primary_zeros_before_fir_samples
        ):
            raise ValueError("primary pre-FIR delay와 TrainingTimingContract가 다릅니다")
        if int(self.secondary_operator.coarse_delay_samples) != int(
            timing.secondary_delay_samples
        ):
            raise ValueError("secondary delay와 TrainingTimingContract가 다릅니다")
        peak = int(np.abs(self.primary_operator.post_onset_fir).argmax())
        if peak != int(timing.primary_fir_peak_offset_samples):
            raise ValueError("primary FIR peak와 TrainingTimingContract가 다릅니다")
        effective = int(self.primary_operator.coarse_delay_samples) + peak
        if effective != int(timing.primary_effective_delay_samples):
            raise ValueError("primary effective delay와 TrainingTimingContract가 다릅니다")
        expected_bands = tuple(
            tuple(float(value) for value in band)
            for band in canonical.physical_identification_subbands_hz
        )
        actual_bands = tuple(
            tuple(float(value) for value in band)
            for band in self.verified_physical_subbands_hz
        )
        if actual_bands != expected_bands:
            raise ValueError(
                "operator가 canonical 88.388--11,313.708 Hz physical subband 전체를 "
                "검증했다는 exact binding이 필요합니다"
            )
        for name, value in (
            ("raw capture SHA", self.raw_capture_sha256),
            ("analysis SHA", self.analysis_sha256),
            ("primary raw capture SHA", self.primary_raw_capture_sha256),
            ("secondary raw capture SHA", self.secondary_raw_capture_sha256),
            ("primary analysis SHA", self.primary_analysis_sha256),
            ("secondary analysis SHA", self.secondary_analysis_sha256),
            ("plant authority SHA", self.plant_authority_sha256),
            ("electrical witness receipt SHA", self.electrical_witness_receipt_sha256),
            ("ERR channel selection SHA", self.err_channel_selection_sha256),
            ("reference channel selection SHA", self.reference_channel_selection_sha256),
        ):
            _require_sha256(value, label=name)
        if (
            self.primary_raw_capture_sha256 != self.raw_capture_sha256
            or self.secondary_raw_capture_sha256 != self.raw_capture_sha256
        ):
            raise ValueError("primary/secondary operator가 같은 immutable raw capture에 결속되지 않았습니다")
        if (
            self.primary_analysis_sha256 != self.analysis_sha256
            or self.secondary_analysis_sha256 != self.analysis_sha256
        ):
            raise ValueError("primary/secondary operator가 같은 immutable analysis에 결속되지 않았습니다")
        for label, value in (
            ("ERR channel index", self.err_channel_index),
            ("reference channel index", self.reference_channel_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label}는 0 이상 int여야 합니다")
        if self.primary_operator.authority_sha256 != self.plant_authority_sha256:
            raise ValueError("primary operator가 같은 plant authority에 결속되지 않았습니다")
        if self.secondary_operator.authority_sha256 != self.plant_authority_sha256:
            raise ValueError("secondary operator가 같은 plant authority에 결속되지 않았습니다")

    @property
    def fixture_only(self) -> bool:
        return bool(self._fixture_only)

    @property
    def secondary_history_samples(self) -> int:
        return int(self.secondary_operator.history_samples)

    @property
    def required_prefix_samples(self) -> int:
        """P/S 모두의 과거가 zero-reset segment 안에 들어오게 하는 최소 history."""

        return max(
            int(self.primary_operator.history_samples),
            int(self.secondary_operator.history_samples),
        )

    @property
    def canonical_training_eligible(self) -> bool:
        return False

    @property
    def training_blocker(self) -> str:
        return FULL_OCTAVE_CAUSAL_PREFIX_TRAINING_BLOCKER

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "authority_schema": self.authority_schema,
            "control_band_contract": self.control_band_contract.model_dump(mode="json"),
            "control_band_contract_sha256": self.control_band_contract_sha256,
            "training_timing_contract": self.training_timing_contract.model_dump(
                mode="json"
            ),
            "training_timing_contract_sha256": self.training_timing_contract_sha256,
            "primary_operator": _operator_payload(self.primary_operator),
            "secondary_operator": _operator_payload(self.secondary_operator),
            "verified_physical_subbands_hz": [
                [float(lo), float(hi)] for lo, hi in self.verified_physical_subbands_hz
            ],
            "raw_capture_sha256": self.raw_capture_sha256,
            "analysis_sha256": self.analysis_sha256,
            "primary_raw_capture_sha256": self.primary_raw_capture_sha256,
            "secondary_raw_capture_sha256": self.secondary_raw_capture_sha256,
            "primary_analysis_sha256": self.primary_analysis_sha256,
            "secondary_analysis_sha256": self.secondary_analysis_sha256,
            "plant_authority_sha256": self.plant_authority_sha256,
            "electrical_witness_receipt_sha256": self.electrical_witness_receipt_sha256,
            "err_channel_index": int(self.err_channel_index),
            "err_channel_selection_sha256": self.err_channel_selection_sha256,
            "reference_channel_index": int(self.reference_channel_index),
            "reference_channel_selection_sha256": self.reference_channel_selection_sha256,
            "block_size": int(self.block_size),
            "fixture_only": self.fixture_only,
            "canonical_training_eligible": False,
            "training_blocker": self.training_blocker,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_payload())).hexdigest()
