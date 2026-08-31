"""Canonical recorded-addition source gain의 무출력 계획 계약.

이 모듈은 오디오 장치를 열지 않는다. exact source-plan과 source bytes를 reference
amplitude에서 렌더링하고, strict primary FIR을 적용한 ERR 예측으로 각 source의
안전한 amplitude 구간을 계산한다.

schema-v1은 ERR operator만 가져 canonical live를 열지 않는다. schema-v2는 bounded
다중레벨 raw에서 독립 holdout을 통과한 NS→ERR/REF safety operator를 exact source마다
적용해 peak/RMS와 residual margin을 계산한다. 이 operator는 recording gain safety
전용이며 ANC P/S·lead·성능 authority로 승격하지 않는다.
"""

from __future__ import annotations

import csv
import copy
import contextlib
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy import signal

from deep_anc.data.holdout_contract import read_regular_file_snapshot
from deep_anc.data.repository_fd import RepositoryFileGuard
from deep_anc.data.recording_source_preflight import (
    SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS,
    SOURCE_PREFLIGHT_FRAMES,
    SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO,
    SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE,
    SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED,
    SOURCE_PREFLIGHT_SAMPLE_RATE,
    rendered_source_preflight,
    timeline_source_feasibility,
)
from deep_anc.data.recording_gain_linearity import (
    ADC_CERTIFICATION_PEAK,
    GAIN_LINEARITY_AUTHORITY_SCOPE,
    GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA,
    OPERATOR_FIR_LENGTH,
    OPERATOR_PEAK_PRE_ROLL_SAMPLES,
    OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO,
    RecordingGainLinearityError,
    validate_gain_linearity_receipt,
)
from deep_anc.realtime.noise_gen import NoiseProgram, render_recording_file_window
RECORDING_SOURCE_GAIN_SCHEMA = "recording_source_gain_plan/v1"
RECORDING_SOURCE_GAIN_SCHEMA_V2 = (
    "recording_source_gain_plan/v3_dynamic_gainprobe006"
)
RECORDING_SOURCE_GAIN_SESSION_BINDING_SCHEMA = (
    "recording_source_gain_session_binding/v3_dynamic_gainprobe006"
)
# schema-v1의 0.06은 이미 발행된 진단용 plan을 재검증하기 위한 legacy 기준이다.
# schema-v2는 이 값을 reference/외삽 기준으로 사용하지 않고 physical receipt가
# receipt가 인증한 dynamic amplitude(독립 0.006 holdout 이하)를 reference/hard cap으로 쓴다.
LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS = 60_000
MINIMUM_AMPLITUDE_MILLIONTHS = 1
ADC_PEAK_HARD_CEILING = 0.5
# 별도 RMS 물리 authority가 생기기 전에는 peak hard ceiling과 같은 linear ceiling을
# 사용한다. peak가 항상 더 강한 조건이지만 RMS 예측과 제한 원인을 독립 보존한다.
ADC_RMS_HARD_CEILING = ADC_PEAK_HARD_CEILING
GAIN_REQUIRED_BANDS_HZ = ((150.0, 600.0), (600.0, 1600.0))
GAIN_PLAN_BLOCKERS = (
    "missing_reference_channel_upper_authority",
    "missing_multilevel_linearity_receipt",
    "strict_primary_not_authoritative_above_1600_hz",
    "recording_level_campaign_v2_not_implemented",
)
PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS = 6_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_DUCT_CONFIG = "configs/duct.yaml"
CANONICAL_HARDWARE_CONFIG = "configs/hardware_jetson.yaml"

# DNS/DEMAND selector의 isolated ``--help`` 경계는 CUDA/torch 없이도 import되어야
# 한다. strict-P planner를 실제 호출할 때만 DSP/runtime 모듈을 가져온다. 이 네
# 이름은 테스트가 synthetic full-provenance validator로 교체할 수도 있다.
load_secondary_path: Any = None
PlantDelays: Any = None
TrainingTimingContract: Any = None
validate_runtime_plant_contract: Any = None


class RecordingSourceGainError(ValueError):
    """Source gain plan이 immutable/안전 계약을 만족하지 않는다."""


def _strict_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    global load_secondary_path
    global PlantDelays
    global TrainingTimingContract
    global validate_runtime_plant_contract
    if load_secondary_path is None:
        from deep_anc.dsp.secondary_path import load_secondary_path as loader

        load_secondary_path = loader
    if PlantDelays is None or TrainingTimingContract is None:
        from deep_anc.dsp.timing import PlantDelays as delays_type
        from deep_anc.dsp.timing import TrainingTimingContract as timing_type

        PlantDelays = delays_type
        TrainingTimingContract = timing_type
    if validate_runtime_plant_contract is None:
        from deep_anc.realtime.plant_contract import (
            validate_runtime_plant_contract as validator,
        )

        validate_runtime_plant_contract = validator
    return (
        load_secondary_path,
        PlantDelays,
        TrainingTimingContract,
        validate_runtime_plant_contract,
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _repo_root(value: str | Path) -> Path:
    return Path(value).resolve()


def _relative_path(value: str | Path, *, label: str) -> str:
    text = Path(value).as_posix()
    path = Path(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or text != path.as_posix()
    ):
        raise RecordingSourceGainError(f"{label}는 canonical 저장소 상대경로여야 합니다")
    return text


def _snapshot(
    repo_root: Path, relative: str, *, label: str, capture_bytes: bool = True
):
    relative = _relative_path(relative, label=label)
    return read_regular_file_snapshot(
        repo_root / relative,
        root=repo_root,
        label=label,
        capture_bytes=capture_bytes,
    )


def _file_ref(relative: str, snapshot) -> dict[str, Any]:
    return {
        "path": relative,
        "size": int(snapshot.size),
        "sha256": str(snapshot.sha256),
    }


def _require_sha(value: str, *, label: str) -> str:
    text = str(value).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise RecordingSourceGainError(f"{label}는 소문자 SHA-256이어야 합니다")
    return text


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingSourceGainError(f"{label}는 finite 숫자여야 합니다")
    number = float(value)
    if not math.isfinite(number):
        raise RecordingSourceGainError(f"{label}는 finite 숫자여야 합니다")
    return number


def _validate_strict_primary_authority(
    repo_root: Path, relative: str, expected_sha256: str
) -> dict[str, Any]:
    """Canonical duct의 full strict P/S raw·analysis provenance를 재검증한다.

    작은 NPZ 하나가 ``fir/delay/band`` scalar만 흉내 내 source-gain authority가 되는
    것을 막는다. 기존 realtime strict-plant validator를 absolute repo-local 경로로
    호출하므로 P/S metadata, same capture, observed PCM, raw/analysis SHA, level evidence,
    lead까지 같은 경계에서 검증된다. 이 함수는 operator를 ANC plant로 새로 승격하지
    않고 이미 canonical duct가 지정한 authority인지 확인만 한다.
    """

    try:
        (
            strict_loader,
            delays_type,
            timing_type,
            plant_validator,
        ) = _strict_runtime_dependencies()
        with contextlib.ExitStack() as stack:
            duct_guard = stack.enter_context(
                RepositoryFileGuard(
                    repo_root, CANONICAL_DUCT_CONFIG, label="canonical duct config"
                )
            )
            hardware_guard = stack.enter_context(
                RepositoryFileGuard(
                    repo_root,
                    CANONICAL_HARDWARE_CONFIG,
                    label="canonical hardware config",
                )
            )
            duct = yaml.safe_load(duct_guard.bytes)
            hardware = yaml.safe_load(hardware_guard.bytes)
            if not isinstance(duct, dict) or not isinstance(hardware, dict):
                raise RecordingSourceGainError("canonical duct/hardware YAML mapping이 아닙니다")
            digital = duct.get("digital_reference") or {}
            secondary_cfg = duct.get("secondary_path") or {}
            strict_cfg = duct.get("strict_measurement") or {}
            configured_primary = _relative_path(
                str(digital.get("primary_path_npz", "")),
                label="duct digital primary",
            )
            if relative != configured_primary:
                raise RecordingSourceGainError(
                    "strict primary가 canonical configs/duct.yaml digital path가 아닙니다"
                )
            secondary_relative = _relative_path(
                str(secondary_cfg.get("npz", "")), label="duct strict secondary"
            )
            level_relative = _relative_path(
                str(strict_cfg.get("measurement_level_evidence", "")),
                label="duct measurement level evidence",
            )
            primary_guard = stack.enter_context(
                RepositoryFileGuard(repo_root, relative, label="canonical strict primary")
            )
            secondary_guard = stack.enter_context(
                RepositoryFileGuard(
                    repo_root, secondary_relative, label="canonical strict secondary"
                )
            )
            level_guard = stack.enter_context(
                RepositoryFileGuard(
                    repo_root, level_relative, label="measurement level evidence"
                )
            )
            expected = _require_sha(
                expected_sha256, label="strict primary expected SHA"
            )
            if primary_guard.sha256 != expected:
                raise RecordingSourceGainError(
                    "canonical strict primary SHA가 외부 anchor와 다릅니다"
                )
            primary_path = Path(os.path.abspath(repo_root / relative))
            secondary_path = Path(os.path.abspath(repo_root / secondary_relative))
            level_path = Path(os.path.abspath(repo_root / level_relative))
            primary = strict_loader(primary_path)
            secondary = strict_loader(secondary_path)
            delays = delays_type.from_config(
                duct_cfg=duct,
                primary_delay_samples=int(primary.delay_samples),
                secondary_delay_samples=int(secondary.delay_samples),
                sample_rate=int(primary.sample_rate),
            )
            timing = timing_type.derive(
                primary_fir=primary.fir, plant_delays=delays
            )
            absolute_duct = copy.deepcopy(duct)
            absolute_duct["digital_reference"]["primary_path_npz"] = str(primary_path)
            absolute_duct["secondary_path"]["npz"] = str(secondary_path)
            absolute_duct["strict_measurement"][
                "measurement_level_evidence"
            ] = str(level_path)
            contract = plant_validator(
                {
                    "reference": "digital",
                    "controller": "dl",
                    "hop": int((hardware.get("audio") or {}).get("block_size", 0)),
                    "digital_reference_lead_samples": int(
                        timing.digital_reference_lead_samples
                    ),
                    "hardware": hardware,
                    "duct": absolute_duct,
                }
            )
            for guard in (
                duct_guard,
                hardware_guard,
                primary_guard,
                secondary_guard,
                level_guard,
            ):
                guard.verify()
            duct_snapshot = type(
                "GuardSnapshot",
                (),
                {
                    "size": duct_guard.size,
                    "sha256": duct_guard.sha256,
                },
            )()
            hardware_snapshot = type(
                "GuardSnapshot",
                (),
                {
                    "size": hardware_guard.size,
                    "sha256": hardware_guard.sha256,
                },
            )()
            actual_primary_sha = primary_guard.sha256
            secondary_guard_sha = secondary_guard.sha256
            level_guard_sha = level_guard.sha256
    except (
        OSError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        raise RecordingSourceGainError(
            f"canonical strict P/S raw/analysis provenance 검증 실패: {exc}"
        ) from exc
    if contract is None or contract.primary_path_sha256 != actual_primary_sha:
        raise RecordingSourceGainError("canonical strict plant validator가 P authority를 반환하지 않았습니다")
    if (
        contract.secondary_path_sha256 != secondary_guard_sha
        or contract.measurement_level_evidence_sha256 != level_guard_sha
    ):
        raise RecordingSourceGainError(
            "canonical strict validator 결과가 held S/level authority bytes와 다릅니다"
        )
    return {
        "duct_config": _file_ref(CANONICAL_DUCT_CONFIG, duct_snapshot),
        "hardware_config": _file_ref(CANONICAL_HARDWARE_CONFIG, hardware_snapshot),
        "secondary_path": {
            "path": secondary_relative,
            "sha256": contract.secondary_path_sha256,
        },
        "measurement_level_evidence": {
            "path": level_relative,
            "sha256": contract.measurement_level_evidence_sha256,
        },
        "capture_id": contract.capture_id,
        "raw_measurement_sha256": contract.raw_measurement_sha256,
        "analysis_sha256": contract.analysis_sha256,
        "derived_lead_samples": int(timing.digital_reference_lead_samples),
    }


def _load_strict_primary(
    repo_root: Path,
    relative: str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    relative = _relative_path(relative, label="strict primary path")
    authority = _validate_strict_primary_authority(
        repo_root, relative, expected_sha256
    )
    snapshot = _snapshot(repo_root, relative, label="strict primary", capture_bytes=True)
    expected = _require_sha(expected_sha256, label="strict primary expected SHA")
    if snapshot.sha256 != expected:
        raise RecordingSourceGainError("strict primary 외부 SHA와 실제 bytes가 다릅니다")
    assert snapshot.data is not None
    try:
        with np.load(io.BytesIO(snapshot.data), allow_pickle=False) as archive:
            fir_raw = np.asarray(archive["fir"])
            fir = np.asarray(fir_raw, dtype=np.float64).reshape(-1)
            sample_rate = int(np.asarray(archive["sample_rate"]).item())
            delay_samples = int(np.asarray(archive["delay_samples"]).item())
            band = np.asarray(archive["consistency_band_hz"], dtype=np.float64)
            capture_id = str(np.asarray(archive["capture_id"]).item())
            output_channel = str(np.asarray(archive["output_channel"]).item())
            amplitude = float(np.asarray(archive["amplitude"]).item())
            xrun_count = int(np.asarray(archive["xrun_count"]).item())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RecordingSourceGainError(
            f"strict primary NPZ를 읽을 수 없습니다: {relative}: {exc}"
        ) from exc
    if (
        fir_raw.ndim != 1
        or fir.size < 2
        or not bool(np.isfinite(fir).all())
        or sample_rate != SOURCE_PREFLIGHT_SAMPLE_RATE
        or delay_samples < 0
        or band.tolist() != [150.0, 1600.0]
        or not capture_id
        or output_channel != "noise"
        or not math.isclose(amplitude, 0.003, rel_tol=0.0, abs_tol=1e-12)
        or xrun_count != 0
    ):
        raise RecordingSourceGainError(
            "strict primary sample-rate/FIR/band/channel/level/xrun 계약이 다릅니다"
        )
    canonical_fir = np.ascontiguousarray(fir, dtype="<f4")
    evidence = {
        **_file_ref(relative, snapshot),
        "capture_id": capture_id,
        "sample_rate": sample_rate,
        "delay_samples": delay_samples,
        "trusted_band_hz": [150.0, 1600.0],
        "measurement_amplitude": amplitude,
        "fir_samples": int(fir.size),
        "fir_float32_le_sha256": _sha256_bytes(canonical_fir.tobytes()),
        "canonical_authority": authority,
    }
    return evidence, fir


def _read_source_rows(
    repo_root: Path,
    relative: str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relative = _relative_path(relative, label="source plan path")
    snapshot = _snapshot(repo_root, relative, label="source plan", capture_bytes=True)
    expected = _require_sha(expected_sha256, label="source plan expected SHA")
    if snapshot.sha256 != expected:
        raise RecordingSourceGainError("source plan 외부 SHA와 실제 bytes가 다릅니다")
    assert snapshot.data is not None
    try:
        text = snapshot.data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = tuple(reader.fieldnames or ())
        raw_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise RecordingSourceGainError(f"source plan CSV를 읽을 수 없습니다: {exc}") from exc
    required = {
        "path",
        "seconds",
        "start_seconds",
        "source_file_sha256",
        "source_family",
        "group_id",
        "lineage_key",
        "split",
    }
    if not required.issubset(fields) or not raw_rows:
        raise RecordingSourceGainError(
            "source plan은 필수 열과 하나 이상의 source row를 가져야 합니다"
        )
    rows: list[dict[str, Any]] = []
    for row_number, raw in enumerate(raw_rows, start=2):
        path = _relative_path(raw.get("path", ""), label=f"source row {row_number} path")
        try:
            seconds = float(raw.get("seconds", ""))
            start_seconds = float(raw.get("start_seconds", ""))
        except (TypeError, ValueError) as exc:
            raise RecordingSourceGainError(
                f"source row {row_number} seconds/start가 숫자가 아닙니다"
            ) from exc
        if (
            not math.isfinite(seconds)
            or not math.isclose(seconds, 15.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isfinite(start_seconds)
            or start_seconds < 0.0
        ):
            raise RecordingSourceGainError(
                f"source row {row_number}는 exact 15초와 0 이상 start를 가져야 합니다"
            )
        source_sha = _require_sha(
            raw.get("source_file_sha256", ""),
            label=f"source row {row_number} SHA",
        )
        source_snapshot = _snapshot(
            repo_root, path, label=f"source row {row_number} audio", capture_bytes=True
        )
        if source_snapshot.sha256 != source_sha:
            raise RecordingSourceGainError(
                f"source row {row_number} source bytes가 plan SHA와 다릅니다"
            )
        assert source_snapshot.data is not None
        identity = {field: str(raw.get(field, "")) for field in fields}
        rows.append(
            {
                "source_row_number": row_number,
                "path": path,
                "seconds": seconds,
                "start_seconds": start_seconds,
                "source_file": _file_ref(path, source_snapshot),
                "source_identity_sha256": _sha256_bytes(
                    _canonical_json_bytes(identity)
                ),
                "source_bytes": source_snapshot.data,
            }
        )
    return _file_ref(relative, snapshot), rows


def _render(row: Mapping[str, Any], amplitude_millionths: int) -> np.ndarray:
    if (
        isinstance(amplitude_millionths, bool)
        or not isinstance(amplitude_millionths, int)
        or not MINIMUM_AMPLITUDE_MILLIONTHS
        <= amplitude_millionths
        <= LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingSourceGainError("amplitude_millionths가 canonical 범위 밖입니다")
    amplitude = float(amplitude_millionths) / 1_000_000.0
    try:
        program = NoiseProgram(
            {
                "type": "file",
                "file": str(row["path"]),
                "file_start_seconds": float(row["start_seconds"]),
                "amplitude": amplitude,
            },
            SOURCE_PREFLIGHT_SAMPLE_RATE,
            file_bytes=bytes(row["source_bytes"]),
        )
        rendered = render_recording_file_window(
            program,
            SOURCE_PREFLIGHT_FRAMES,
            sample_rate=SOURCE_PREFLIGHT_SAMPLE_RATE,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecordingSourceGainError(
            f"source row {row['source_row_number']} exact render 실패: {exc}"
        ) from exc
    values = np.ascontiguousarray(rendered, dtype="<f4")
    if values.shape != (SOURCE_PREFLIGHT_FRAMES,) or not bool(np.isfinite(values).all()):
        raise RecordingSourceGainError("exact rendered source shape/finite 계약 위반")
    return values


def _band_levels(samples: np.ndarray) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = int(values.size)
    window = np.hanning(n)
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(n, 1.0 / SOURCE_PREFLIGHT_SAMPLE_RATE)
    denominator = n * float(np.sum(window**2))
    result: dict[str, float] = {}
    for low, high in GAIN_REQUIRED_BANDS_HZ:
        selected = (frequencies >= low) & (frequencies <= high)
        power = 2.0 * float(np.sum(np.abs(spectrum[selected]) ** 2)) / denominator
        result[f"{int(low)}_{int(high)}"] = 10.0 * math.log10(max(power, 1e-24))
    return result


def _err_prediction(source: np.ndarray, fir: np.ndarray) -> dict[str, Any]:
    values = np.asarray(source, dtype=np.float64)
    predicted = signal.fftconvolve(values, np.asarray(fir, dtype=np.float64), mode="full")
    if not bool(np.isfinite(predicted).all()):
        raise RecordingSourceGainError("strict-P predicted ERR에 NaN/Inf가 있습니다")
    peak = float(np.max(np.abs(predicted)))
    rms = float(math.sqrt(float(np.sum(np.square(predicted))) / values.size))
    floor = np.finfo(np.float64).tiny
    return {
        "frames_in": int(values.size),
        "frames_full_convolution": int(predicted.size),
        "peak_linear": peak,
        "peak_dbfs": float(20.0 * math.log10(max(peak, floor))),
        "rms_linear": rms,
        "rms_dbfs": float(20.0 * math.log10(max(rms, floor))),
        "band_metric": "hann_full_convolution_rfft_v1",
        "band_rms_dbfs": _band_levels(predicted),
    }


def _required_snr_db() -> float:
    coherence = SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
    return float(10.0 * math.log10(coherence / (1.0 - coherence)))


def _lower_from_db(
    reference_db: float, required_db: float, *, reference_amplitude_millionths: int
) -> int:
    value = reference_amplitude_millionths * 10.0 ** (
        (required_db - reference_db) / 20.0
    )
    return max(MINIMUM_AMPLITUDE_MILLIONTHS, int(math.ceil(value - 1e-12)))


def _timeline_lower(
    reference: np.ndarray, *, reference_amplitude_millionths: int
) -> int:
    low = MINIMUM_AMPLITUDE_MILLIONTHS
    high = reference_amplitude_millionths

    def passes(value: int) -> bool:
        scaled = np.asarray(
            np.asarray(reference, dtype=np.float64)
            * (float(value) / reference_amplitude_millionths),
            dtype=np.float32,
        )
        return timeline_source_feasibility(scaled)["passed"] is True

    if not passes(high):
        return high + 1
    while low < high:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _upper_from_linear(
    reference_value: float,
    ceiling: float,
    *,
    reference_amplitude_millionths: int,
) -> int:
    if not reference_value > 0.0:
        raise RecordingSourceGainError("predicted physical upper 값이 0입니다")
    value = reference_amplitude_millionths * float(ceiling) / reference_value
    return min(
        reference_amplitude_millionths,
        max(0, int(math.floor(value + 1e-12))),
    )


def _physical_operator_contract(analysis: Any) -> dict[str, Any]:
    """독립 검증된 receipt에서 gain-safety 전용 두 operator를 엄격히 읽는다."""

    operator = analysis.get("safety_operator") if isinstance(analysis, Mapping) else None
    if not isinstance(operator, Mapping):
        raise RecordingSourceGainError("gain-linearity safety operator가 없습니다")
    supported = analysis.get("supported_max_amplitude_millionths")
    tested = analysis.get("tested_max_amplitude_millionths")
    if (
        isinstance(supported, bool)
        or not isinstance(supported, int)
        or isinstance(tested, bool)
        or not isinstance(tested, int)
        or not MINIMUM_AMPLITUDE_MILLIONTHS
        <= supported
        <= tested
        == PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingSourceGainError(
            "gain-linearity tested/supported dynamic cap 계약 위반"
        )
    if (
        analysis.get("distortion_certified") is not False
        or analysis.get("physical_authority_scope")
        != GAIN_LINEARITY_AUTHORITY_SCOPE
    ):
        raise RecordingSourceGainError(
            "gain-linearity authority는 tested ADC peak safety 전용이어야 합니다"
        )
    unsealed = dict(operator)
    seal = unsealed.pop("operator_sha256", None)
    if operator.get("schema") == GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA:
        if (
            operator.get("role")
            != "source_gain_peak_envelope_only_not_anc_plant_authority"
            or operator.get("fit_levels_millionths") != [3_000, 4_000, 5_000]
            or operator.get("independent_holdout_level_millionths") != tested
            or operator.get("tested_max_amplitude_millionths") != tested
            or operator.get("supported_max_amplitude_millionths") != supported
            or operator.get("complex_operator_thresholds_relaxed") is not False
            or operator.get("complex_operator_used_as_authority") is not False
            or not isinstance(seal, str)
            or _SHA256_RE.fullmatch(seal) is None
            or seal != _seal(unsealed)
        ):
            raise RecordingSourceGainError(
                "gain-linearity peak envelope seal/role 불일치"
            )
        channels = operator.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != {"err", "ref"}:
            raise RecordingSourceGainError(
                "gain-linearity peak envelope ERR/REF 집합 불일치"
            )
        result: dict[str, Any] = {}
        for name in ("err", "ref"):
            item = channels[name]
            gain_upper = _finite(
                item.get("peak_gain_upper_with_uncertainty")
                if isinstance(item, Mapping)
                else None,
                label=f"{name} peak envelope gain",
            )
            if (
                not isinstance(item, Mapping)
                or gain_upper <= 0.0
                or item.get("uncertainty_factor") != 1.25
                or item.get("valid_through_amplitude_millionths") != tested
                or item.get("prediction")
                != "upper_peak=gain_upper*rendered_source_peak"
            ):
                raise RecordingSourceGainError(
                    f"gain-linearity {name} peak envelope 계약 위반"
                )
            result[name] = {"peak_gain_upper_with_uncertainty": gain_upper}
        return {
            "operator_sha256": str(seal),
            "role": str(operator["role"]),
            "prediction_kind": "measured_peak_envelope",
            "tested_max_amplitude_millionths": tested,
            "supported_max_amplitude_millionths": supported,
            "distortion_certified": False,
            "physical_authority_scope": GAIN_LINEARITY_AUTHORITY_SCOPE,
            "channels": result,
        }
    if (
        operator.get("schema")
        != "recording_gain_safety_operator/v3_gainprobe006"
        or operator.get("role")
        != "source_gain_prediction_only_not_anc_plant_authority"
        or operator.get("fir_length") != OPERATOR_FIR_LENGTH
        or operator.get("peak_pre_roll_samples")
        != OPERATOR_PEAK_PRE_ROLL_SAMPLES
        or operator.get("relative_subband_minimum_target_norm_ratio")
        != OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO
        or not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _seal(unsealed)
    ):
        raise RecordingSourceGainError("gain-linearity safety operator seal/role 불일치")
    channels = operator.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != {"err", "ref"}:
        raise RecordingSourceGainError("gain-linearity ERR/REF operator 집합 불일치")
    result: dict[str, Any] = {}
    for name in ("err", "ref"):
        item = channels[name]
        if not isinstance(item, Mapping) or item.get("passed") is not True:
            raise RecordingSourceGainError(f"gain-linearity {name} operator가 PASS가 아닙니다")
        raw_fir = item.get("fir")
        if not isinstance(raw_fir, list) or len(raw_fir) != OPERATOR_FIR_LENGTH:
            raise RecordingSourceGainError(f"gain-linearity {name} FIR 길이 불일치")
        fir = np.asarray(raw_fir, dtype=np.float32)
        residual = item.get("residual_bound")
        if not isinstance(residual, Mapping):
            raise RecordingSourceGainError(f"gain-linearity {name} residual bound 누락")
        induced_l1 = _finite(
            residual.get("induced_fir_l1_upper"), label=f"{name} induced L1"
        )
        absolute_peak = _finite(
            residual.get("unexplained_peak_absolute_upper"),
            label=f"{name} unexplained peak",
        )
        absolute_rms = _finite(
            residual.get("unexplained_rms_absolute_upper"),
            label=f"{name} unexplained RMS",
        )
        if (
            item.get("fir_encoding") != "float32_le"
            or not bool(np.isfinite(fir).all())
            or _sha256_bytes(np.ascontiguousarray(fir, dtype="<f4").tobytes())
            != item.get("fir_sha256")
            or residual.get("definition")
            != "young_l1_induced_plus_measured_absolute_with_uncertainty_v1"
            or residual.get("valid_through_amplitude_millionths") != tested
            or induced_l1 < 0.0
            or absolute_peak < 0.0
            or absolute_rms < 0.0
        ):
            raise RecordingSourceGainError(f"gain-linearity {name} FIR/margin 계약 위반")
        result[name] = {
            "fir": np.asarray(fir, dtype=np.float64),
            "fir_sha256": str(item["fir_sha256"]),
            "valid_through_amplitude_millionths": tested,
            "induced_fir_l1_upper": induced_l1,
            "unexplained_peak_absolute_upper": absolute_peak,
            "unexplained_rms_absolute_upper": absolute_rms,
        }
    return {
        "operator_sha256": str(seal),
        "role": str(operator["role"]),
        "prediction_kind": "compact_fir_with_residual",
        "tested_max_amplitude_millionths": tested,
        "supported_max_amplitude_millionths": supported,
        "distortion_certified": False,
        "physical_authority_scope": GAIN_LINEARITY_AUTHORITY_SCOPE,
        "channels": result,
    }


def _physical_prediction(
    source: np.ndarray, operator: Mapping[str, Any]
) -> dict[str, Any]:
    """Exact rendered source에 ERR/REF compact operator와 residual margin을 적용한다."""

    values = np.asarray(source, dtype=np.float64)
    source_peak = float(np.max(np.abs(values)))
    source_rms = math.sqrt(float(np.mean(np.square(values))))
    result: dict[str, Any] = {}
    for name in ("err", "ref"):
        channel = operator["channels"][name]
        if operator.get("prediction_kind") == "measured_peak_envelope":
            upper = source_peak * float(
                channel["peak_gain_upper_with_uncertainty"]
            )
            result[name] = {
                "prediction_kind": "measured_peak_envelope",
                "operator_fir_sha256": None,
                "model_peak_linear": 0.0,
                "model_rms_linear": 0.0,
                "residual_peak_margin_linear": upper,
                # RMS cannot exceed peak; use the same upper rather than inventing
                # an unmeasured RMS transfer function.
                "residual_rms_margin_linear": upper,
                "upper_peak_linear": upper,
                "upper_rms_linear": upper,
            }
            continue
        predicted = signal.fftconvolve(values, channel["fir"], mode="full")
        model_peak = float(np.max(np.abs(predicted)))
        model_rms = math.sqrt(float(np.sum(np.square(predicted))) / values.size)
        residual_peak = (
            source_peak * float(channel["induced_fir_l1_upper"])
            + float(channel["unexplained_peak_absolute_upper"])
        )
        residual_rms = (
            source_rms * float(channel["induced_fir_l1_upper"])
            + float(channel["unexplained_rms_absolute_upper"])
        )
        result[name] = {
            "operator_fir_sha256": str(channel["fir_sha256"]),
            "model_peak_linear": model_peak,
            "model_rms_linear": model_rms,
            "residual_peak_margin_linear": residual_peak,
            "residual_rms_margin_linear": residual_rms,
            "upper_peak_linear": model_peak + residual_peak,
            "upper_rms_linear": model_rms + residual_rms,
        }
    return result


def _row_gain_evidence(
    row: Mapping[str, Any],
    fir: np.ndarray,
    *,
    reference_amplitude_millionths: int = LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS,
    physical_operator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        isinstance(reference_amplitude_millionths, bool)
        or not isinstance(reference_amplitude_millionths, int)
        or not MINIMUM_AMPLITUDE_MILLIONTHS
        <= reference_amplitude_millionths
        <= LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingSourceGainError("source gain reference amplitude가 canonical 범위 밖입니다")
    reference = _render(row, reference_amplitude_millionths)
    reference_preflight = rendered_source_preflight(reference)
    reference_prediction = _err_prediction(reference, fir)
    required_snr = _required_snr_db()
    required_err_band = (
        SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS + required_snr
    )
    band_lowers = {
        band: _lower_from_db(
            level,
            required_err_band,
            reference_amplitude_millionths=reference_amplitude_millionths,
        )
        for band, level in reference_prediction["band_rms_dbfs"].items()
    }
    source_trusted_lower = _lower_from_db(
        float(reference_preflight["trusted_band_rms_dbfs"]),
        float(reference_preflight["minimum_trusted_band_rms_dbfs"]),
        reference_amplitude_millionths=reference_amplitude_millionths,
    )
    timeline_lower = _timeline_lower(
        reference,
        reference_amplitude_millionths=reference_amplitude_millionths,
    )
    lower_reasons = {
        "source_trusted_level": source_trusted_lower,
        "timeline_feasibility": timeline_lower,
        **{f"err_snr_{band}": value for band, value in band_lowers.items()},
    }
    upper_reasons = {
        "err_peak": _upper_from_linear(
            float(reference_prediction["peak_linear"]),
            ADC_PEAK_HARD_CEILING,
            reference_amplitude_millionths=reference_amplitude_millionths,
        ),
        "err_rms": _upper_from_linear(
            float(reference_prediction["rms_linear"]),
            ADC_RMS_HARD_CEILING,
            reference_amplitude_millionths=reference_amplitude_millionths,
        ),
        "reference_amplitude": reference_amplitude_millionths,
    }
    reference_physical: dict[str, Any] | None = None
    if physical_operator is not None:
        reference_physical = _physical_prediction(reference, physical_operator)
        upper_reasons.update(
            {
                "physical_err_peak_with_residual": _upper_from_linear(
                    float(reference_physical["err"]["upper_peak_linear"]),
                    ADC_CERTIFICATION_PEAK,
                    reference_amplitude_millionths=reference_amplitude_millionths,
                ),
                "physical_err_rms_with_residual": _upper_from_linear(
                    float(reference_physical["err"]["upper_rms_linear"]),
                    ADC_RMS_HARD_CEILING,
                    reference_amplitude_millionths=reference_amplitude_millionths,
                ),
                "physical_ref_peak_with_residual": _upper_from_linear(
                    float(reference_physical["ref"]["upper_peak_linear"]),
                    ADC_CERTIFICATION_PEAK,
                    reference_amplitude_millionths=reference_amplitude_millionths,
                ),
                "physical_ref_rms_with_residual": _upper_from_linear(
                    float(reference_physical["ref"]["upper_rms_linear"]),
                    ADC_RMS_HARD_CEILING,
                    reference_amplitude_millionths=reference_amplitude_millionths,
                ),
                "physical_probe_supported_max": int(
                    physical_operator["supported_max_amplitude_millionths"]
                ),
            }
        )
    lower = max(lower_reasons.values())
    upper = min(upper_reasons.values())
    if lower > upper:
        raise RecordingSourceGainError(
            f"source row {row['source_row_number']} gain feasible interval이 없습니다: "
            f"lower={lower}, upper={upper}"
        )

    # 최대 SNR을 유지하는 가장 큰 exact micro-amplitude를 선택한다. float32 rounding이
    # analytic upper를 1 ulp 넘을 수 있으므로 exact render/strict-P 재검산으로만 확정한다.
    selected = int(upper)
    final_source: np.ndarray | None = None
    final_preflight: dict[str, Any] | None = None
    final_prediction: dict[str, Any] | None = None
    final_band_snr: dict[str, float] | None = None
    final_physical: dict[str, Any] | None = None
    final_ok = False
    # analytic scaling은 float64이지만 실제 renderer는 float32다. ceiling 바로 위의
    # 1-ulp overshoot를 통과시키지 않고 micro-amplitude를 아래로만 최대 32칸 조정한다.
    for candidate in range(selected, max(lower - 1, selected - 32), -1):
        source_candidate = _render(row, candidate)
        preflight_candidate = rendered_source_preflight(source_candidate)
        prediction_candidate = _err_prediction(source_candidate, fir)
        snr_candidate = {
            band: float(level - SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS)
            for band, level in prediction_candidate["band_rms_dbfs"].items()
        }
        physical_candidate = (
            _physical_prediction(source_candidate, physical_operator)
            if physical_operator is not None
            else None
        )
        physical_ok = bool(
            physical_candidate is None
            or all(
                float(physical_candidate[name]["upper_peak_linear"])
                <= ADC_CERTIFICATION_PEAK
                and float(physical_candidate[name]["upper_rms_linear"])
                <= ADC_RMS_HARD_CEILING
                for name in ("err", "ref")
            )
        )
        candidate_ok = bool(
            candidate >= lower
            and preflight_candidate["passed"] is True
            and float(prediction_candidate["peak_linear"])
            <= ADC_PEAK_HARD_CEILING
            and float(prediction_candidate["rms_linear"])
            <= ADC_RMS_HARD_CEILING
            and all(value >= required_snr for value in snr_candidate.values())
            and physical_ok
        )
        if candidate_ok:
            selected = candidate
            final_source = source_candidate
            final_preflight = preflight_candidate
            final_prediction = prediction_candidate
            final_band_snr = snr_candidate
            final_physical = physical_candidate
            final_ok = True
            break
    if not final_ok:
        raise RecordingSourceGainError(
            f"source row {row['source_row_number']} selected exact gain 재검산이 FAIL입니다"
        )
    assert final_source is not None
    assert final_preflight is not None
    assert final_prediction is not None
    assert final_band_snr is not None
    selected_bytes = np.ascontiguousarray(final_source, dtype="<f4").tobytes()
    return {
        "source_row_number": int(row["source_row_number"]),
        "source_identity_sha256": str(row["source_identity_sha256"]),
        "source_file": dict(row["source_file"]),
        "reference_amplitude_millionths": reference_amplitude_millionths,
        "reference_render_sample_sha256": str(reference_preflight["sample_sha256"]),
        "reference_source_preflight": reference_preflight,
        "reference_predicted_err": reference_prediction,
        "reference_physical_prediction": reference_physical,
        "bounds": {
            "lower_amplitude_millionths": int(lower),
            "upper_amplitude_millionths": int(upper),
            "lower_constraints": lower_reasons,
            "upper_constraints": upper_reasons,
        },
        "selected_amplitude_millionths": selected,
        "selected_amplitude": float(selected) / 1_000_000.0,
        "selected_render_sample_sha256": _sha256_bytes(selected_bytes),
        "selected_source_preflight": final_preflight,
        "selected_predicted_err": final_prediction,
        "selected_predicted_err_snr_db": final_band_snr,
        "selected_physical_prediction": final_physical,
        "feasible": True,
    }


def _source_cap_evidence(
    row: Mapping[str, Any],
    fir: np.ndarray,
    *,
    amplitude_millionths: int,
) -> dict[str, Any]:
    """Physical receipt의 measured cap에서 selector 필요조건만 계산한다.

    이 결과는 ERR/REF 물리 상한이나 live 권위가 아니다. exact source가 timeline,
    source level, strict-P 150--1600 Hz SNR을 만족하지 못하면 Elice bundle을 발행해도
    v2 source-gain plan이 될 수 없으므로 그 왕복을 미리 막는 용도다.
    """

    source = _render(row, amplitude_millionths)
    preflight = rendered_source_preflight(source)
    predicted = _err_prediction(source, fir)
    required_snr = _required_snr_db()
    snr = {
        band: float(level - SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS)
        for band, level in predicted["band_rms_dbfs"].items()
    }
    reasons: list[str] = []
    if preflight["passed"] is not True:
        reasons.append("rendered_source_preflight")
    for band, value in sorted(snr.items()):
        if value < required_snr:
            reasons.append(f"strict_primary_snr_{band}")
    if float(predicted["peak_linear"]) > ADC_PEAK_HARD_CEILING:
        reasons.append("strict_primary_peak")
    if float(predicted["rms_linear"]) > ADC_RMS_HARD_CEILING:
        reasons.append("strict_primary_rms")
    return {
        "source_row_number": int(row["source_row_number"]),
        "source_identity_sha256": str(row["source_identity_sha256"]),
        "source_file": dict(row["source_file"]),
        "start_seconds": float(row["start_seconds"]),
        "amplitude_millionths": int(amplitude_millionths),
        "rendered_source_preflight": preflight,
        "strict_primary_prediction": predicted,
        "strict_primary_snr_db": snr,
        "minimum_predicted_snr_db": required_snr,
        "feasible": not reasons,
        "blocker_reasons": reasons,
    }


def audit_source_plan_at_measured_cap(
    *,
    repo_root: str | Path,
    source_plan: str,
    expected_source_plan_sha256: str,
    strict_primary: str,
    expected_strict_primary_sha256: str,
    amplitude_millionths: int,
) -> dict[str, Any]:
    """19행 selector output을 물리 probe cap에서 무출력 fail-closed 감사한다."""

    if (
        isinstance(amplitude_millionths, bool)
        or not isinstance(amplitude_millionths, int)
        or not MINIMUM_AMPLITUDE_MILLIONTHS
        <= amplitude_millionths
        <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingSourceGainError(
            "selector amplitude는 probe tested max 0.006 이하여야 합니다"
        )
    root = _repo_root(repo_root)
    source_ref, rows = _read_source_rows(
        root, source_plan, expected_sha256=expected_source_plan_sha256
    )
    strict_ref, fir = _load_strict_primary(
        root, strict_primary, expected_sha256=expected_strict_primary_sha256
    )
    evidence = [
        _source_cap_evidence(
            row, fir, amplitude_millionths=amplitude_millionths
        )
        for row in rows
    ]
    blockers = [
        {
            "source_row_number": item["source_row_number"],
            "path": item["source_file"]["path"],
            "reasons": list(item["blocker_reasons"]),
        }
        for item in evidence
        if item["feasible"] is not True
    ]
    payload: dict[str, Any] = {
        "schema": "recording_source_selector_cap_audit/v1",
        "role": "selector_precondition_only_not_live_gain_or_anc_plant_authority",
        "source_plan": source_ref,
        "strict_primary": strict_ref,
        "amplitude_millionths": amplitude_millionths,
        "row_count": len(evidence),
        "feasible_row_count": len(evidence) - len(blockers),
        "all_rows_feasible": not blockers,
        "blockers": blockers,
        "rows": evidence,
    }
    payload["evidence_sha256"] = _seal(payload)
    return payload


def select_best_feasible_source_window(
    *,
    row: Mapping[str, Any],
    strict_primary_fir: np.ndarray,
    candidate_start_seconds: Sequence[float],
    amplitude_millionths: int = PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS,
) -> dict[str, Any]:
    """유한 후보 grid를 exact gate/결정론적 score로 전수 평가한다.

    score는 두 strict-P band 중 낮은 SNR, timeline eligible ratio, source trusted
    level 순이다. 완전 동률이면 더 이른 start를 선택한다. 수기 후보 교체가 아니라
    caller가 봉인한 candidate grid 전체에 같은 규칙을 적용하도록 제공한다.
    """

    starts = sorted({_finite(value, label="candidate start") for value in candidate_start_seconds})
    if not starts or starts[0] < 0.0:
        raise RecordingSourceGainError("candidate start grid가 비었거나 음수입니다")
    candidates: list[dict[str, Any]] = []
    for start in starts:
        candidate = dict(row)
        candidate["start_seconds"] = start
        identity = {
            "path": str(candidate["path"]),
            "seconds": float(candidate["seconds"]),
            "start_seconds": start,
            "source_file_sha256": str(candidate["source_file"]["sha256"]),
        }
        candidate["source_identity_sha256"] = _sha256_bytes(
            _canonical_json_bytes(identity)
        )
        item = _source_cap_evidence(
            candidate,
            strict_primary_fir,
            amplitude_millionths=amplitude_millionths,
        )
        candidates.append(item)
    feasible = [item for item in candidates if item["feasible"] is True]
    if not feasible:
        raise RecordingSourceGainError("candidate grid에 feasible source window가 없습니다")

    def score(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        return (
            min(float(value) for value in item["strict_primary_snr_db"].values()),
            float(
                item["rendered_source_preflight"]["timeline_feasibility"][
                    "eligible_ratio"
                ]
            ),
            float(item["rendered_source_preflight"]["trusted_band_rms_dbfs"]),
            -float(item["start_seconds"]),
        )

    selected = max(feasible, key=score)
    return {
        "schema": "recording_source_window_selection/v1",
        "amplitude_millionths": amplitude_millionths,
        "candidate_starts": starts,
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "selected_start_seconds": float(selected["start_seconds"]),
        "selected_evidence": selected,
        "score_definition": (
            "max_min_strict_primary_snr_then_timeline_then_trusted_then_earliest"
        ),
    }


def build_recording_source_gain_plan(
    *,
    repo_root: str | Path,
    source_plan: str,
    expected_source_plan_sha256: str,
    strict_primary: str,
    expected_strict_primary_sha256: str,
    gain_linearity_receipt: str | None = None,
    expected_gain_linearity_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Exact source/strict-P bytes에서 deterministic gain plan을 만든다."""

    root = _repo_root(repo_root)
    source_plan_ref, rows = _read_source_rows(
        root,
        source_plan,
        expected_sha256=expected_source_plan_sha256,
    )
    strict_ref, fir = _load_strict_primary(
        root,
        strict_primary,
        expected_sha256=expected_strict_primary_sha256,
    )
    if (gain_linearity_receipt is None) != (
        expected_gain_linearity_receipt_sha256 is None
    ):
        raise RecordingSourceGainError(
            "gain-linearity receipt path/SHA는 함께 필요합니다"
        )
    physical_summary: dict[str, Any] | None = None
    physical_ref: dict[str, Any] | None = None
    physical_hardware: dict[str, Any] | None = None
    physical_operator: dict[str, Any] | None = None
    physical_source_commit: str | None = None
    if gain_linearity_receipt is not None:
        try:
            physical_summary = validate_gain_linearity_receipt(
                repo_root=root,
                receipt_path=gain_linearity_receipt,
                expected_sha256=str(expected_gain_linearity_receipt_sha256),
            )
        except RecordingGainLinearityError as exc:
            raise RecordingSourceGainError(
                f"gain-linearity receipt 검증 실패: {exc}"
            ) from exc
        if physical_summary.get("passed") is not True:
            raise RecordingSourceGainError("FAIL gain-linearity receipt는 사용할 수 없습니다")
        physical_payload = physical_summary["payload"]
        receipt_hardware = physical_payload.get("hardware")
        fingerprint = (
            receipt_hardware.get("physical_fingerprint")
            if isinstance(receipt_hardware, Mapping)
            else None
        )
        if (
            not isinstance(receipt_hardware, Mapping)
            or not isinstance(fingerprint, Mapping)
            or not {"path", "size", "sha256", "physical_fingerprint_sha256"}.issubset(
                receipt_hardware
            )
            or receipt_hardware.get("physical_fingerprint_sha256")
            != _seal(fingerprint)
        ):
            raise RecordingSourceGainError(
                "gain-linearity receipt hardware/fingerprint binding이 불완전합니다"
            )
        physical_hardware = json.loads(_canonical_json_bytes(dict(receipt_hardware)))
        analysis = physical_payload.get("analysis")
        if physical_payload.get("analysis", {}).get(
            "safety_operator_is_anc_plant_authority"
        ) is not False:
            raise RecordingSourceGainError(
                "gain-linearity safety operator가 ANC plant authority와 분리되지 않았습니다"
            )
        physical_operator = _physical_operator_contract(analysis)
        if (
            not isinstance(analysis, Mapping)
            or analysis.get("supported_max_amplitude_millionths")
            != physical_operator["supported_max_amplitude_millionths"]
            or analysis.get("tested_max_amplitude_millionths")
            != physical_operator["tested_max_amplitude_millionths"]
        ):
            raise RecordingSourceGainError(
                "gain-linearity measured supported max와 operator bound가 다릅니다"
            )
        physical_source_commit = str(physical_payload.get("source_commit", ""))
        if re.fullmatch(r"[0-9a-f]{40}", physical_source_commit) is None:
            raise RecordingSourceGainError("gain-linearity source commit 형식 불일치")
        receipt_snapshot = _snapshot(
            root,
            gain_linearity_receipt,
            label="gain-linearity receipt",
            capture_bytes=False,
        )
        physical_ref = _file_ref(gain_linearity_receipt, receipt_snapshot)
    reference_amplitude_millionths = (
        int(physical_operator["supported_max_amplitude_millionths"])
        if physical_operator is not None
        else LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS
    )
    row_evidence = [
        _row_gain_evidence(
            row,
            fir,
            reference_amplitude_millionths=reference_amplitude_millionths,
            physical_operator=physical_operator,
        )
        for row in rows
    ]
    contract = {
        "sample_rate": SOURCE_PREFLIGHT_SAMPLE_RATE,
        "frames": SOURCE_PREFLIGHT_FRAMES,
        "reference_amplitude_millionths": reference_amplitude_millionths,
        "minimum_amplitude_millionths": MINIMUM_AMPLITUDE_MILLIONTHS,
        "adc_peak_hard_ceiling": ADC_PEAK_HARD_CEILING,
        "adc_rms_hard_ceiling": ADC_RMS_HARD_CEILING,
        "required_bands_hz": [list(item) for item in GAIN_REQUIRED_BANDS_HZ],
        "conservative_quiet_floor_dbfs": (
            SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS
        ),
        "required_capture_coherence_squared": (
            SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
        ),
        "minimum_predicted_snr_db": _required_snr_db(),
        "minimum_timeline_eligible_ratio": SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO,
        "source_reference_amplitude": (
            float(reference_amplitude_millionths) / 1_000_000.0
        ),
        "legacy_schema_v1_source_reference_amplitude": (
            SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE
            if physical_operator is None
            else None
        ),
        "amplitude_encoding": "integer_millionths",
        "prediction_authority": (
            (
                "strict_primary_for_150_1600_snr_plus_measured_err_ref_peak_envelope"
                if physical_operator.get("prediction_kind")
                == "measured_peak_envelope"
                else "strict_primary_for_150_1600_snr_plus_gain_safety_err_ref_operator"
            )
            if physical_operator is not None
            else "strict_primary_err_only"
        ),
        "safety_operator_is_anc_plant_authority": False,
        "distortion_certified": False,
        "physical_authority_scope": GAIN_LINEARITY_AUTHORITY_SCOPE,
    }
    if physical_operator is not None:
        contract["gain_linearity_source_commit"] = physical_source_commit
        contract["safety_operator_sha256"] = physical_operator["operator_sha256"]
        contract["physical_probe_tested_max_amplitude_millionths"] = (
            physical_operator["tested_max_amplitude_millionths"]
        )
    live_eligible = physical_summary is not None
    payload: dict[str, Any] = {
        "schema": (
            RECORDING_SOURCE_GAIN_SCHEMA_V2
            if live_eligible
            else RECORDING_SOURCE_GAIN_SCHEMA
        ),
        "status": (
            "READY_PHYSICAL_GAIN_BOUND"
            if live_eligible
            else "BLOCKED_PENDING_REF_AND_LINEARITY_AUTHORITY"
        ),
        "canonical_live_eligible": live_eligible,
        "blocker_reasons": [] if live_eligible else list(GAIN_PLAN_BLOCKERS),
        "source_plan": source_plan_ref,
        "strict_primary": strict_ref,
        "contract": contract,
        "row_count": len(row_evidence),
        "rows": row_evidence,
    }
    if live_eligible:
        payload["gain_linearity_receipt"] = physical_ref
        payload["gain_linearity_hardware"] = physical_hardware
    payload["evidence_sha256"] = _seal(payload)
    return payload


def _read_plan_payload(
    *, repo_root: Path, plan_path: str, expected_sha256: str
) -> tuple[dict[str, Any], Any]:
    relative = _relative_path(plan_path, label="source gain plan path")
    snapshot = _snapshot(repo_root, relative, label="source gain plan", capture_bytes=True)
    expected = _require_sha(expected_sha256, label="source gain plan expected SHA")
    if snapshot.sha256 != expected:
        raise RecordingSourceGainError("source gain plan 외부 SHA와 실제 bytes가 다릅니다")
    assert snapshot.data is not None
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingSourceGainError(f"source gain plan JSON 오류: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecordingSourceGainError("source gain plan payload가 mapping이 아닙니다")
    return payload, snapshot


def validate_recording_source_gain_plan(
    *,
    repo_root: str | Path,
    plan_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Plan bytes와 결속된 source/strict-P를 모두 다시 계산한다."""

    root = _repo_root(repo_root)
    payload, snapshot = _read_plan_payload(
        repo_root=root, plan_path=plan_path, expected_sha256=expected_sha256
    )
    seal = payload.get("evidence_sha256")
    unsealed = dict(payload)
    unsealed.pop("evidence_sha256", None)
    if (
        not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _seal(unsealed)
    ):
        raise RecordingSourceGainError("source gain plan self-seal이 다릅니다")
    source_ref = payload.get("source_plan")
    strict_ref = payload.get("strict_primary")
    physical_ref = payload.get("gain_linearity_receipt")
    physical_hardware = payload.get("gain_linearity_hardware")
    if (
        not isinstance(source_ref, Mapping)
        or set(source_ref) != {"path", "size", "sha256"}
        or not isinstance(strict_ref, Mapping)
        or not {"path", "size", "sha256"}.issubset(strict_ref)
    ):
        raise RecordingSourceGainError("source gain plan file ref schema가 다릅니다")
    schema = payload.get("schema")
    if schema not in {RECORDING_SOURCE_GAIN_SCHEMA, RECORDING_SOURCE_GAIN_SCHEMA_V2}:
        raise RecordingSourceGainError("source gain plan schema가 다릅니다")
    if schema == RECORDING_SOURCE_GAIN_SCHEMA_V2:
        if not isinstance(physical_ref, Mapping) or set(physical_ref) != {
            "path",
            "size",
            "sha256",
        }:
            raise RecordingSourceGainError("v2 gain-linearity receipt ref schema 위반")
        if not isinstance(physical_hardware, Mapping):
            raise RecordingSourceGainError("v2 gain-linearity hardware binding 누락")
    elif physical_ref is not None or physical_hardware is not None:
        raise RecordingSourceGainError("v1 source gain plan에 physical receipt가 있습니다")
    rebuilt = build_recording_source_gain_plan(
        repo_root=root,
        source_plan=str(source_ref["path"]),
        expected_source_plan_sha256=str(source_ref["sha256"]),
        strict_primary=str(strict_ref["path"]),
        expected_strict_primary_sha256=str(strict_ref["sha256"]),
        gain_linearity_receipt=(
            str(physical_ref["path"]) if isinstance(physical_ref, Mapping) else None
        ),
        expected_gain_linearity_receipt_sha256=(
            str(physical_ref["sha256"]) if isinstance(physical_ref, Mapping) else None
        ),
    )
    if payload != rebuilt:
        raise RecordingSourceGainError(
            "source gain plan이 source/strict-P 독립 재계산과 다릅니다"
        )
    return {
        "plan_path": _relative_path(plan_path, label="source gain plan path"),
        "plan_size": int(snapshot.size),
        "plan_sha256": str(snapshot.sha256),
        "payload": payload,
        "canonical_live_eligible": payload.get("canonical_live_eligible") is True,
    }


def build_recording_source_gain_session_binding(
    summary: Mapping[str, Any],
    *,
    source_row_number: int,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Validated v2 plan의 exact row/amplitude를 live session용으로 축약한다."""

    payload = summary.get("payload") if isinstance(summary, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or summary.get("canonical_live_eligible") is not True
        or payload.get("schema") != RECORDING_SOURCE_GAIN_SCHEMA_V2
        or payload.get("canonical_live_eligible") is not True
    ):
        raise RecordingSourceGainError("canonical-live v2 source gain summary가 아닙니다")
    commit = str(expected_source_commit).lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RecordingSourceGainError("expected source commit이 exact SHA가 아닙니다")
    contract = payload.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("gain_linearity_source_commit") != commit
        or contract.get("safety_operator_is_anc_plant_authority") is not False
        or contract.get("distortion_certified") is not False
        or contract.get("physical_authority_scope")
        != GAIN_LINEARITY_AUTHORITY_SCOPE
    ):
        raise RecordingSourceGainError(
            "source gain plan commit 또는 non-plant safety role이 current execution과 다릅니다"
        )
    if isinstance(source_row_number, bool) or not isinstance(source_row_number, int):
        raise RecordingSourceGainError("source row number는 int여야 합니다")
    rows = payload.get("rows")
    matches = [
        row
        for row in rows if isinstance(row, Mapping)
        and row.get("source_row_number") == source_row_number
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise RecordingSourceGainError("source gain plan exact row가 하나가 아닙니다")
    row = matches[0]
    millionths = row.get("selected_amplitude_millionths")
    measured_cap = contract.get("reference_amplitude_millionths")
    if (
        row.get("feasible") is not True
        or isinstance(millionths, bool)
        or not isinstance(millionths, int)
        or not MINIMUM_AMPLITUDE_MILLIONTHS
        <= millionths
        <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
        or isinstance(measured_cap, bool)
        or not isinstance(measured_cap, int)
        or not millionths
        <= measured_cap
        <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
        or not isinstance(row.get("selected_physical_prediction"), Mapping)
    ):
        raise RecordingSourceGainError("source gain row가 feasible physical bound가 아닙니다")
    source_plan = payload.get("source_plan")
    receipt = payload.get("gain_linearity_receipt")
    hardware = payload.get("gain_linearity_hardware")
    if (
        not isinstance(source_plan, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(hardware, Mapping)
    ):
        raise RecordingSourceGainError("source/linearity file ref가 없습니다")
    binding: dict[str, Any] = {
        "schema": RECORDING_SOURCE_GAIN_SESSION_BINDING_SCHEMA,
        "source_gain_plan": {
            "path": summary.get("plan_path"),
            "size": summary.get("plan_size"),
            "sha256": summary.get("plan_sha256"),
        },
        "source_plan_sha256": source_plan.get("sha256"),
        "gain_linearity_receipt": dict(receipt),
        "gain_linearity_hardware": dict(hardware),
        "source_commit": commit,
        "safety_operator_sha256": contract.get("safety_operator_sha256"),
        "safety_operator_is_anc_plant_authority": False,
        "distortion_certified": False,
        "physical_authority_scope": GAIN_LINEARITY_AUTHORITY_SCOPE,
        "source_row_number": source_row_number,
        "source_identity_sha256": row.get("source_identity_sha256"),
        "source_file": dict(row.get("source_file") or {}),
        "amplitude_millionths": millionths,
        "amplitude": millionths / 1_000_000.0,
        "supported_max_amplitude_millionths": measured_cap,
        "render_sample_sha256": row.get("selected_render_sample_sha256"),
        "physical_prediction": row.get("selected_physical_prediction"),
    }
    binding["binding_sha256"] = _seal(binding)
    return binding


def validate_recording_source_gain_session_binding(
    summary: Mapping[str, Any], binding: Any
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise RecordingSourceGainError("source gain session binding이 mapping이 아닙니다")
    seal = binding.get("binding_sha256")
    unsealed = dict(binding)
    unsealed.pop("binding_sha256", None)
    if (
        binding.get("schema") != RECORDING_SOURCE_GAIN_SESSION_BINDING_SCHEMA
        or not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _seal(unsealed)
    ):
        raise RecordingSourceGainError("source gain session binding seal/schema 불일치")
    rebuilt = build_recording_source_gain_session_binding(
        summary,
        source_row_number=binding.get("source_row_number"),
        expected_source_commit=str(binding.get("source_commit", "")),
    )
    if dict(binding) != rebuilt:
        raise RecordingSourceGainError("source gain session binding 독립 재유도 불일치")
    return json.loads(_canonical_json_bytes(dict(binding)))


def issue_recording_source_gain_plan(
    *,
    repo_root: str | Path,
    output_path: str,
    source_plan: str,
    expected_source_plan_sha256: str,
    strict_primary: str,
    expected_strict_primary_sha256: str,
    gain_linearity_receipt: str | None = None,
    expected_gain_linearity_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Gain plan을 no-replace로 발행하고 외부 SHA를 반환한다."""

    root = _repo_root(repo_root)
    relative = _relative_path(output_path, label="source gain output path")
    payload = build_recording_source_gain_plan(
        repo_root=root,
        source_plan=source_plan,
        expected_source_plan_sha256=expected_source_plan_sha256,
        strict_primary=strict_primary,
        expected_strict_primary_sha256=expected_strict_primary_sha256,
        gain_linearity_receipt=gain_linearity_receipt,
        expected_gain_linearity_receipt_sha256=(
            expected_gain_linearity_receipt_sha256
        ),
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _pretty_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o664)
    except FileExistsError as exc:
        raise RecordingSourceGainError(
            f"source gain plan은 no-replace입니다: {relative}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # 생성 도중 실패한 partial file도 forensic 대상으로 남겨 덮어쓰지 않는다.
        raise
    digest = _sha256_bytes(data)
    return validate_recording_source_gain_plan(
        repo_root=root,
        plan_path=relative,
        expected_sha256=digest,
    )


__all__ = [
    "ADC_PEAK_HARD_CEILING",
    "ADC_RMS_HARD_CEILING",
    "GAIN_PLAN_BLOCKERS",
    "GAIN_REQUIRED_BANDS_HZ",
    "LEGACY_REFERENCE_AMPLITUDE_MILLIONTHS",
    "MINIMUM_AMPLITUDE_MILLIONTHS",
    "PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS",
    "RECORDING_SOURCE_GAIN_SCHEMA",
    "RECORDING_SOURCE_GAIN_SCHEMA_V2",
    "RECORDING_SOURCE_GAIN_SESSION_BINDING_SCHEMA",
    "RecordingSourceGainError",
    "audit_source_plan_at_measured_cap",
    "build_recording_source_gain_plan",
    "build_recording_source_gain_session_binding",
    "issue_recording_source_gain_plan",
    "select_best_feasible_source_window",
    "validate_recording_source_gain_plan",
    "validate_recording_source_gain_session_binding",
]
