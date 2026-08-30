"""실시간 raw의 연속 acoustic pilot로 ADC↔DAC time-map을 조건부 감사한다.

이 모듈은 :mod:`deep_anc.dsp.fullband_causal_v4`의 actual-int16 연속
파일럿과 clock estimator를 재사용한다. 측정용 P/S authority를 발행하지 않으며,
runtime ``source → ERR/REF``의 시간축만 별도 schema로 판정한다.

중요한 한계가 있다. acoustic raw만으로는 ``ADC clock motion``과 ``두 마이크에
공통인 time-varying plant delay``를 정보론적으로 구분할 수 없다. 따라서 모든
수치가 통과해도 독립 clock PASS나 deployment PASS를 발행하지 않는다. 실제
sounddevice capture이며 fixed-LTI stationarity가 통과한 경우에만
``CONDITIONAL_PASS``이고, synthetic fixture는 항상 ``FIXTURE_ONLY_PASS``다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..audio_io import float32_to_pcm_int16
from ..dsp import fullband_causal_v4 as causal_v4
from .clock_telemetry import payload_sha256, sha256_file


RUNTIME_PHYSICAL_WITNESS_PLAN_SCHEMA = "runtime_physical_clock_witness_plan_v1"
RUNTIME_PHYSICAL_WITNESS_SCHEMA = "runtime_physical_clock_witness_v1"
RUNTIME_PHYSICAL_WITNESS_BUNDLE_SCHEMA = (
    "runtime_physical_clock_witness_bundle_v1"
)

MINIMUM_WITNESS_SECONDS = 30.0
MINIMUM_PERIOD_COUNT = math.ceil(
    MINIMUM_WITNESS_SECONDS * causal_v4.FS / causal_v4.PERIOD
)
MAX_ROW_GAIN_DEVIATION_DB = 0.5
_HEX = frozenset("0123456789abcdef")
_FIXED_LTI_SCOPE = (
    "fixed_lti_acoustic_only; common time-varying plant delay remains "
    "observationally equivalent to ADC clock motion"
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{name}는 lowercase SHA-256이어야 합니다")
    return digest


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _fit_period_indices(period_count: int) -> tuple[int, ...]:
    """capture 전에 고정된 시간-분산 fit row를 만든다."""

    if period_count < MINIMUM_PERIOD_COUNT:
        raise ValueError(
            "runtime physical witness는 30초 이상, "
            f"최소 {MINIMUM_PERIOD_COUNT} period가 필요합니다"
        )
    candidates = (
        0,
        1,
        period_count // 3,
        min(period_count - 1, period_count // 3 + 1),
        (2 * period_count) // 3,
        min(period_count - 1, (2 * period_count) // 3 + 1),
        period_count - 2,
        period_count - 1,
    )
    result = tuple(sorted(set(int(value) for value in candidates)))
    if len(result) < 6:
        raise AssertionError("runtime witness fit row가 시간축을 덮지 못합니다")
    return result


def build_runtime_physical_witness_plan(
    *,
    session_npz_target: str | Path,
    clock_receipt_target: str | Path,
    hardware_fingerprint_sha256: str,
    analysis_start_sample: int,
    synthetic_fixture: bool,
    expected_input_device_prefix: str = "APE:1",
    expected_output_device_prefix: str = "Audio:0",
    period_count: int = MINIMUM_PERIOD_COUNT,
) -> dict[str, Any]:
    """actual capture 전에 저장할 fail-closed runtime witness plan을 만든다.

    ``analysis_start_sample``은 파일럿 phase origin에 정확히 맞춰야 하며
    결과를 보고 탐색하지 않는다. 현 schema는 단순화를 위해 v4
    period 배수만 허용한다.
    """

    if type(synthetic_fixture) is not bool:
        raise ValueError("synthetic_fixture는 명시적 bool이어야 합니다")
    start = int(analysis_start_sample)
    count = int(period_count)
    if start < 0 or start % causal_v4.PERIOD:
        raise ValueError(
            "analysis_start_sample은 0 이상의 continuous-pilot period 배수여야 합니다"
        )
    fit_indices = _fit_period_indices(count)
    validation_indices = tuple(
        index for index in range(count) if index not in set(fit_indices)
    )
    if not validation_indices:
        raise ValueError("runtime witness validation row가 비었습니다")

    pilot = causal_v4.continuous_pilot_period()
    pilot_bins = causal_v4._pilot_bin_sets()["primary"]
    frequencies = np.fft.rfftfreq(causal_v4.PERIOD, 1.0 / causal_v4.FS)[
        pilot_bins
    ]
    if not (
        float(np.min(frequencies)) >= causal_v4.PILOT_BAND[0]
        and float(np.max(frequencies)) <= causal_v4.PILOT_BAND[1]
    ):
        raise AssertionError("runtime witness가 v4 저역 pilot 밖을 사용합니다")

    session_target = _resolved(session_npz_target)
    clock_target = _resolved(clock_receipt_target)
    if session_target == clock_target:
        raise ValueError("session NPZ와 clock receipt target은 다른 파일이어야 합니다")
    if not expected_input_device_prefix.strip() or not expected_output_device_prefix.strip():
        raise ValueError("expected runtime device prefix가 비었습니다")

    rows = []
    for index in range(count):
        row_start = start + index * causal_v4.PERIOD
        rows.append(
            {
                "name": f"runtime_pilot_{index:03d}",
                "start_frame": row_start,
                "stop_frame": row_start + causal_v4.PERIOD,
                "frames": causal_v4.PERIOD,
                "purpose": "fit" if index in fit_indices else "validation",
                "role": "runtime_source_to_err_ref",
            }
        )

    plan: dict[str, Any] = {
        "schema_version": RUNTIME_PHYSICAL_WITNESS_PLAN_SCHEMA,
        "capture_origin": (
            "synthetic_fixture" if synthetic_fixture else "physical_sounddevice_runtime"
        ),
        "synthetic_fixture": synthetic_fixture,
        "sample_rate": causal_v4.FS,
        "block_size": causal_v4.BLOCK,
        "analysis_start_sample": start,
        "analysis_stop_sample": start + count * causal_v4.PERIOD,
        "period_count": count,
        "observed_seconds": count * causal_v4.PERIOD / causal_v4.FS,
        "minimum_observed_seconds": MINIMUM_WITNESS_SECONDS,
        "session_npz_target": str(session_target),
        "clock_receipt_target": str(clock_target),
        "hardware_fingerprint_sha256": _require_sha256(
            hardware_fingerprint_sha256, "hardware_fingerprint_sha256"
        ),
        "expected_input_device_prefix": str(expected_input_device_prefix),
        "expected_output_device_prefix": str(expected_output_device_prefix),
        "clock_rows": rows,
        "fit_period_indices": list(fit_indices),
        "validation_period_indices": list(validation_indices),
        "continuous_reserved_pilot": {
            "source_output_channel": 0,
            "control_output_channel_must_be_null_on_pilot_lines": True,
            "period_samples": causal_v4.PERIOD,
            "period_pcm_sha256": _array_sha256(pilot),
            "source_period_pcm_sha256": _array_sha256(pilot[:, 0]),
            "pilot_bins": pilot_bins.tolist(),
            "pilot_frequencies_hz": frequencies.tolist(),
            "band_hz": list(causal_v4.PILOT_BAND),
            "source_line_max_complex_error": 1.0e-8,
            "control_line_max_absolute": 1.0e-8,
            "minimum_source_line_magnitude": 1.0e3,
            "actual_int16_required": True,
        },
        "clock_gate": {
            "maximum_abs_ppm": causal_v4.CLOCK_MAX_ABS_PPM,
            "view_end_to_end_disagreement_max_samples": (
                causal_v4.CLOCK_VIEW_DISAGREEMENT_MAX
            ),
            "leaveout_max_samples": causal_v4.CLOCK_LEAVEOUT_MAX,
            "linear_cubic_max_samples": causal_v4.CLOCK_CUBIC_MAX,
            "combined_max_samples": causal_v4.CLOCK_COMBINED_MAX,
            "hard_20db_11314hz_max_samples": causal_v4.CLOCK_HARD_MAX,
            "minimum_transfer_coherence": causal_v4.CLOCK_MIN_COHERENCE,
            "maximum_row_gain_deviation_db": MAX_ROW_GAIN_DEVIATION_DB,
            "minimum_full_anc_gain_seconds": MINIMUM_WITNESS_SECONDS,
        },
        "authority_scope": {
            "highband_target_or_attenuation_used_for_clock_fit": False,
            "pilot_band_is_clock_witness_not_control_or_evaluation_band": True,
            "control_attenuation_assessed": False,
            "octave_125_hz_band_hz": [
                125.0 / math.sqrt(2.0),
                125.0 * math.sqrt(2.0),
            ],
            "octave_125_hz_fully_covered_by_pilot": False,
            "point_control_union_150_11314_claimed_by_witness": False,
            "ns_and_cs_share_one_dac_clock": True,
            "adc_dac_drift_is_not_ns_cs_relative_phase_drift": True,
            "fixed_lti_hypothesis_required": True,
            "independent_electrical_clock_witness_present": False,
            "independent_clock_pass_possible": False,
            "scope_limitation": _FIXED_LTI_SCOPE,
        },
    }
    plan["plan_sha256"] = _json_sha256(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    if payload.get("schema_version") != RUNTIME_PHYSICAL_WITNESS_PLAN_SCHEMA:
        raise ValueError("runtime physical witness plan schema가 다릅니다")
    if type(payload.get("synthetic_fixture")) is not bool:
        raise ValueError("plan synthetic_fixture 명시가 없습니다")
    expected_origin = (
        "synthetic_fixture"
        if payload["synthetic_fixture"]
        else "physical_sounddevice_runtime"
    )
    if payload.get("capture_origin") != expected_origin:
        raise ValueError("capture_origin과 synthetic_fixture가 다릅니다")
    stored_sha = _require_sha256(payload.get("plan_sha256"), "plan_sha256")
    unhashed = dict(payload)
    unhashed.pop("plan_sha256", None)
    if _json_sha256(unhashed) != stored_sha:
        raise ValueError("runtime physical witness plan SHA가 다릅니다")
    if (int(payload.get("sample_rate", 0)), int(payload.get("block_size", 0))) != (
        causal_v4.FS,
        causal_v4.BLOCK,
    ):
        raise ValueError("runtime witness는 48kHz/256 samples여야 합니다")
    count = int(payload.get("period_count", 0))
    if count < MINIMUM_PERIOD_COUNT:
        raise ValueError("runtime witness 관측 시간이 30초 미만입니다")
    if float(payload.get("observed_seconds", 0.0)) < MINIMUM_WITNESS_SECONDS:
        raise ValueError("runtime witness 관측 시간이 30초 미만입니다")
    pilot = causal_v4.continuous_pilot_period()
    pilot_contract = payload.get("continuous_reserved_pilot", {})
    if pilot_contract.get("period_pcm_sha256") != _array_sha256(pilot):
        raise ValueError("runtime witness pilot period SHA가 v4 authority와 다릅니다")
    if pilot_contract.get("source_period_pcm_sha256") != _array_sha256(pilot[:, 0]):
        raise ValueError("runtime witness source pilot SHA가 v4 authority와 다릅니다")
    expected_bins = causal_v4._pilot_bin_sets()["primary"].tolist()
    if pilot_contract.get("pilot_bins") != expected_bins:
        raise ValueError("runtime witness pilot bin이 v4 authority와 다릅니다")
    scope = payload.get("authority_scope", {})
    if scope.get("highband_target_or_attenuation_used_for_clock_fit") is not False:
        raise ValueError("고역 target/감쇠 결과를 clock fit에 사용할 수 없습니다")
    if scope.get("pilot_band_is_clock_witness_not_control_or_evaluation_band") is not True:
        raise ValueError("runtime pilot band를 control/evaluation band로 오인했습니다")
    if scope.get("control_attenuation_assessed") is not False:
        raise ValueError("clock witness는 control attenuation을 판정할 수 없습니다")
    if scope.get("octave_125_hz_fully_covered_by_pilot") is not False:
        raise ValueError("152–600Hz pilot은 125Hz octave 전체를 덮지 않습니다")
    if scope.get("point_control_union_150_11314_claimed_by_witness") is not False:
        raise ValueError("clock witness를 point-control union 증거로 승격할 수 없습니다")
    if scope.get("ns_and_cs_share_one_dac_clock") is not True:
        raise ValueError("NS/CS common DAC clock semantics가 누락됐습니다")
    if scope.get("adc_dac_drift_is_not_ns_cs_relative_phase_drift") is not True:
        raise ValueError("ADC–DAC drift를 NS–CS 상대 drift로 혼동했습니다")
    return payload


def write_runtime_physical_witness_plan_exclusive(
    path: str | Path, plan: Mapping[str, Any]
) -> tuple[Path, str]:
    """predeclared plan을 O_EXCL/no-replace로 저장한다."""

    payload = _validate_plan(plan)
    return _write_json_exclusive(path, payload)


def _write_json_exclusive(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(target, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return target, hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root가 object가 아닙니다: {path}")
    return value


def _validate_ring_backlog(telemetry: Mapping[str, Any]) -> dict[str, int]:
    """SPSC의 정상 one-hop 점유와 실제 초과 backlog를 분리한다.

    callback/engine scheduling 순서에 따라 관측 absolute backlog는 0 또는
    한 block일 수 있다. 따라서 absolute 0을 요구하지 않고, predeclared
    one-hop 허용량 이하이며 그 허용량을 넘은 excess가 exact 0인지 판정한다.
    """

    fields = (
        "maximum_input_backlog_samples",
        "maximum_output_backlog_samples",
        "allowed_input_backlog_samples",
        "allowed_output_backlog_samples",
    )
    values: dict[str, int] = {}
    for field in fields:
        raw = telemetry.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"runtime {field}는 JSON integer여야 합니다")
        if raw < 0:
            raise ValueError(f"runtime {field}는 0 이상이어야 합니다")
        values[field] = raw

    allowed_input = values["allowed_input_backlog_samples"]
    allowed_output = values["allowed_output_backlog_samples"]
    if allowed_input != causal_v4.BLOCK or allowed_output != causal_v4.BLOCK:
        raise ValueError(
            "runtime allowed backlog은 현 256-sample one-hop 계약과 다릅니다"
        )
    maximum_input = values["maximum_input_backlog_samples"]
    maximum_output = values["maximum_output_backlog_samples"]
    excess_input = max(0, maximum_input - allowed_input)
    excess_output = max(0, maximum_output - allowed_output)
    if excess_input != 0 or excess_output != 0:
        raise ValueError(
            "runtime maximum excess backlog이 0이 아닙니다: "
            f"input={excess_input}, output={excess_output}"
        )
    return {
        **values,
        "maximum_excess_input_backlog_samples": excess_input,
        "maximum_excess_output_backlog_samples": excess_output,
    }


def _validate_clock_bundle(
    *, bundle: Mapping[str, Any], session_path: Path, session_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, int]]:
    if bundle.get("schema_version") != "realtime_clock_receipt_bundle_v1":
        raise ValueError("runtime clock receipt bundle schema가 다릅니다")
    if bundle.get("authority_status") != "INCONCLUSIVE":
        raise ValueError("runtime clock telemetry가 구조 PASS/INCONCLUSIVE 상태가 아닙니다")
    if _require_sha256(
        bundle.get("recording_npz_sha256"), "recording_npz_sha256"
    ) != session_sha256:
        raise ValueError("clock receipt의 runtime NPZ SHA가 다릅니다")
    if _resolved(bundle.get("recording_npz", "")) != session_path:
        raise ValueError("clock receipt의 runtime NPZ path가 다릅니다")
    telemetry = bundle.get("runtime_clock_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("runtime clock telemetry payload가 누락됐습니다")
    telemetry = dict(telemetry)
    expected_telemetry_sha = _require_sha256(
        bundle.get("runtime_clock_telemetry_sha256"),
        "runtime_clock_telemetry_sha256",
    )
    if payload_sha256(telemetry) != expected_telemetry_sha:
        raise ValueError("runtime clock telemetry payload SHA가 다릅니다")
    if telemetry.get("schema_version") != "realtime_clock_telemetry_v1":
        raise ValueError("runtime clock telemetry schema가 다릅니다")
    if telemetry.get("authority_status") != "INCONCLUSIVE":
        raise ValueError("runtime clock telemetry authority는 INCONCLUSIVE여야 합니다")
    if telemetry.get("structural_status") != "PASS":
        raise ValueError("runtime clock telemetry structural status가 PASS가 아닙니다")
    clock_semantics = telemetry.get("clock_semantics", {})
    if clock_semantics.get(
        "noise_and_cancel_outputs_share_one_output_stream_device_clock"
    ) is not True:
        raise ValueError("runtime NS/CS common output-clock semantics가 누락됐습니다")
    if clock_semantics.get(
        "adc_dac_drift_is_not_noise_cancel_relative_output_phase"
    ) is not True:
        raise ValueError("runtime ADC–DAC drift를 NS–CS relative drift로 혼동했습니다")

    summary = telemetry.get("callback_summary", {})
    zero_summary = (
        "incomplete_callback_count",
        "pending_callback_count",
        "portaudio_status_callback_count",
        "callback_host_deadline_miss_count",
        "omitted_callback_record_count",
    )
    for field in zero_summary:
        if int(summary.get(field, -1)) != 0:
            raise ValueError(f"runtime callback {field}가 0이 아닙니다")
    callbacks = telemetry.get("callbacks")
    if not isinstance(callbacks, list) or not callbacks:
        raise ValueError("runtime callback raw rows가 누락됐습니다")
    if int(summary.get("callback_count", -1)) != len(callbacks):
        raise ValueError("runtime callback count와 raw row 수가 다릅니다")
    if int(summary.get("stored_callback_record_count", -1)) != len(callbacks):
        raise ValueError("runtime callback raw가 전수 저장되지 않았습니다")
    if int(summary.get("completed_callback_count", -1)) != len(callbacks):
        raise ValueError("runtime completed callback count와 raw row 수가 다릅니다")
    if any(row.get("completed") is not True for row in callbacks):
        raise ValueError("미완료 runtime callback이 있습니다")

    counters = telemetry.get("runtime_counters_final", {})
    zero_counters = (
        "xrun_count",
        "deadline_miss_count",
        "input_ring_drop_samples",
        "output_ring_drop_samples",
        "input_ring_overrun_blocks",
        "output_ring_overrun_blocks",
        "input_ring_underrun_blocks",
        "output_ring_underrun_blocks",
        "ring_add_samples",
        "fallback_silence_blocks",
    )
    for field in zero_counters:
        if int(counters.get(field, -1)) != 0:
            raise ValueError(f"runtime counter {field}가 0이 아닙니다")
    watchdogs = counters.get("watchdog_trip_counts", {})
    if not isinstance(watchdogs, Mapping) or any(int(value) != 0 for value in watchdogs.values()):
        raise ValueError("runtime watchdog trip count가 0이 아닙니다")
    backlog_receipt = _validate_ring_backlog(telemetry)
    if telemetry.get("issue_counts") not in ({}, None):
        raise ValueError("runtime clock telemetry issue count가 0이 아닙니다")
    time_domains = telemetry.get("time_domains", {})
    for name in (
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    ):
        row = time_domains.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"runtime {name} raw summary가 누락됐습니다")
        if int(row.get("finite_count", -1)) != len(callbacks):
            raise ValueError(f"runtime {name} finite count가 callback 수와 다릅니다")
        for field in (
            "missing_or_nonfinite_count",
            "strict_monotonic_violation_count",
            "frame_step_violation_count",
        ):
            if int(row.get(field, -1)) != 0:
                raise ValueError(f"runtime {name} {field}가 0이 아닙니다")

    callback_raw = {
        "frame_index": np.asarray(
            [row["callback_start_frame"] for row in callbacks], dtype=np.int64
        ),
        "frame_count": np.asarray(
            [row["callback_frame_count"] for row in callbacks], dtype=np.int64
        ),
        "input_adc_time": np.asarray(
            [row["input_buffer_adc_time"] for row in callbacks], dtype=np.float64
        ),
        "output_dac_time": np.asarray(
            [row["output_buffer_dac_time"] for row in callbacks], dtype=np.float64
        ),
    }
    return telemetry, callback_raw, backlog_receipt


def _load_session_arrays(
    path: Path, expected_telemetry_sha256: str
) -> dict[str, np.ndarray]:
    required = ("source", "control", "ref", "err", "anc_gain")
    with np.load(path, allow_pickle=False) as archive:
        if any(key not in archive.files for key in required):
            missing = [key for key in required if key not in archive.files]
            raise ValueError(f"runtime session raw array가 누락됐습니다: {missing}")
        fs = int(np.asarray(archive["fs"]).item())
        embedded = str(
            np.asarray(archive["runtime_clock_telemetry_sha256"]).item()
        )
        embedded_status = str(
            np.asarray(archive["runtime_clock_authority_status"]).item()
        )
        arrays = {key: np.asarray(archive[key]) for key in required}
    if fs != causal_v4.FS:
        raise ValueError("runtime session sample rate가 48 kHz가 아닙니다")
    if embedded != expected_telemetry_sha256:
        raise ValueError("runtime NPZ의 embedded clock telemetry SHA가 다릅니다")
    if embedded_status != "INCONCLUSIVE":
        raise ValueError("runtime NPZ의 clock authority status가 INCONCLUSIVE가 아닙니다")
    lengths = {int(value.size) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("runtime session raw array 길이가 다릅니다")
    for key, value in arrays.items():
        if value.dtype != np.float32 or value.ndim != 1:
            raise ValueError(f"runtime {key}는 exact 1-D float32 raw여야 합니다")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"runtime {key}에 non-finite raw가 있습니다")
    return arrays


def _validate_actual_reserved_pilot(
    *, submitted: np.ndarray, plan: Mapping[str, Any]
) -> dict[str, Any]:
    pilot = causal_v4.continuous_pilot_period()
    bins = causal_v4._pilot_bin_sets()["primary"]
    expected = np.fft.rfft(pilot[:, 0].astype(np.float64))[bins]
    contract = plan["continuous_reserved_pilot"]
    max_source_error = 0.0
    max_control = 0.0
    minimum_source = math.inf
    spectra_rows: list[tuple[str, str, str]] = []
    for row in plan["clock_rows"]:
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        window = submitted[start:stop]
        if window.shape != (causal_v4.PERIOD, 2):
            raise ValueError("runtime witness period raw coverage가 부족합니다")
        spectrum = np.fft.rfft(window.astype(np.float64), axis=0)
        source_lines = spectrum[bins, 0]
        control_lines = spectrum[bins, 1]
        source_error = float(np.max(np.abs(source_lines - expected)))
        control_absolute = float(np.max(np.abs(control_lines)))
        source_minimum = float(np.min(np.abs(source_lines)))
        max_source_error = max(max_source_error, source_error)
        max_control = max(max_control, control_absolute)
        minimum_source = min(minimum_source, source_minimum)
        spectra_rows.append(
            (
                str(row["name"]),
                _array_sha256(source_lines),
                _array_sha256(control_lines),
            )
        )
    if max_source_error > float(contract["source_line_max_complex_error"]):
        raise ValueError(
            "actual source PCM이 predeclared reserved pilot line을 exact 보존하지 않습니다"
        )
    if max_control > float(contract["control_line_max_absolute"]):
        raise ValueError(
            "actual control PCM이 source reserved pilot line에 누출되었습니다"
        )
    if minimum_source <= float(contract["minimum_source_line_magnitude"]):
        raise ValueError("actual source reserved pilot denominator가 부족합니다")
    return {
        "maximum_source_line_complex_error": max_source_error,
        "maximum_control_line_absolute": max_control,
        "minimum_source_line_magnitude": minimum_source,
        "segment_spectra_sha256": hashlib.sha256(
            json.dumps(spectra_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "highband_target_or_attenuation_used": False,
    }


def _row_stationarity(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    signals: Mapping[str, np.ndarray],
    rate_ratio: float,
) -> dict[str, Any]:
    bank = causal_v4._transfer_bank(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=rate_ratio,
        method="linear",
        purposes=("fit", "validation"),
        paths=("primary",),
    )
    fit_names = [
        str(row["name"]) for row in plan["clock_rows"] if row["purpose"] == "fit"
    ]
    row_names = [str(row["name"]) for row in plan["clock_rows"]]
    residual_by_view: dict[str, list[float]] = {}
    gain_by_view: dict[str, list[float]] = {}
    coherence_by_view: dict[str, list[float]] = {}
    change_points: list[str] = []
    for microphone in signals:
        reference = np.mean(
            np.stack(
                [bank[(microphone, "primary", name)][0] for name in fit_names],
                axis=0,
            ),
            axis=0,
        )
        frequency = bank[(microphone, "primary", fit_names[0])][1]
        delays: list[float] = []
        gains_db: list[float] = []
        coherences: list[float] = []
        for name in row_names:
            candidate = bank[(microphone, "primary", name)][0]
            floor = max(
                float(np.max(np.abs(reference))),
                float(np.max(np.abs(candidate))),
            ) * 1.0e-8
            valid = (np.abs(reference) > floor) & (np.abs(candidate) > floor)
            if int(np.count_nonzero(valid)) < 8:
                raise ValueError("runtime pilot line SNR/bin count가 부족합니다")
            delay, phase_coherence = causal_v4._fractional_delay(
                candidate[valid] / reference[valid], frequency[valid]
            )
            corrected = candidate[valid] * np.exp(
                2j * np.pi * frequency[valid] * delay / causal_v4.FS
            )
            complex_coherence = float(
                np.abs(np.vdot(reference[valid], corrected))
                / (
                    np.linalg.norm(reference[valid])
                    * np.linalg.norm(corrected)
                    + 1.0e-30
                )
            )
            gain_ratio = float(
                np.linalg.norm(candidate[valid])
                / (np.linalg.norm(reference[valid]) + 1.0e-30)
            )
            gains_db.append(20.0 * math.log10(max(gain_ratio, 1.0e-30)))
            delays.append(float(delay))
            coherences.append(min(float(phase_coherence), complex_coherence))
        residual_by_view[microphone] = delays
        gain_by_view[microphone] = gains_db
        coherence_by_view[microphone] = coherences
        for index, step in enumerate(np.diff(np.asarray(delays))):
            if abs(float(step)) > causal_v4.CLOCK_HARD_MAX:
                change_points.append(f"{microphone}:{index}->{index + 1}:{step}")

    maximum_residual = max(
        abs(value) for values in residual_by_view.values() for value in values
    )
    maximum_gain = max(
        abs(value) for values in gain_by_view.values() for value in values
    )
    minimum_coherence = min(
        value for values in coherence_by_view.values() for value in values
    )
    view_disagreement = max(
        abs(left - right)
        for left, right in zip(residual_by_view["err"], residual_by_view["ref"])
    )
    if maximum_residual > causal_v4.CLOCK_LEAVEOUT_MAX:
        raise ValueError("runtime segment q residual가 fixed-LTI clock 예산을 넘습니다")
    if view_disagreement > causal_v4.CLOCK_VIEW_DISAGREEMENT_MAX:
        raise ValueError("runtime ERR/REF segment q trajectory가 동의하지 않습니다")
    if minimum_coherence < causal_v4.CLOCK_MIN_COHERENCE:
        raise ValueError("runtime reserved pilot transfer coherence가 부족합니다")
    if maximum_gain > MAX_ROW_GAIN_DEVIATION_DB:
        raise ValueError("runtime acoustic pilot gain stationarity가 깨졌습니다")
    if change_points:
        raise ValueError("runtime acoustic q/plant change-point가 감지됐습니다")

    residual_array = np.stack(
        [residual_by_view["err"], residual_by_view["ref"]], axis=0
    ).astype(np.float64)
    slope_array = np.diff(residual_array, axis=1)
    gain_array = np.stack(
        [gain_by_view["err"], gain_by_view["ref"]], axis=0
    ).astype(np.float64)
    coherence_array = np.stack(
        [coherence_by_view["err"], coherence_by_view["ref"]], axis=0
    ).astype(np.float64)
    return {
        "segment_count": len(row_names),
        "segment_names_sha256": hashlib.sha256(
            json.dumps(row_names, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "segment_q_residual_samples": residual_array.tolist(),
        "segment_q_residual_sha256": _array_sha256(residual_array),
        "segment_phase_slope_samples": slope_array.tolist(),
        "segment_phase_slope_sha256": _array_sha256(slope_array),
        "segment_gain_deviation_db_sha256": _array_sha256(gain_array),
        "segment_coherence_sha256": _array_sha256(coherence_array),
        "maximum_absolute_q_residual_samples": float(maximum_residual),
        "maximum_err_ref_q_disagreement_samples": float(view_disagreement),
        "maximum_absolute_gain_deviation_db": float(maximum_gain),
        "minimum_transfer_coherence": float(minimum_coherence),
        "change_point_count": 0,
        "sample_slip_count": 0,
    }


def _run_clock_fit(
    *, plan: Mapping[str, Any], submitted: np.ndarray, err: np.ndarray, ref: np.ndarray
) -> dict[str, Any]:
    # v4와 동일한 저역 전처리와 estimator를 쓴다. 고역 target,
    # ANC attenuation, ERR 성능 결과는 이 경로에 전달되지 않는다.
    sos = butter(12, (120.0, 680.0), btype="bandpass", fs=causal_v4.FS, output="sos")
    signals = {
        "err": sosfiltfilt(sos, np.asarray(err, dtype=np.float64)),
        "ref": sosfiltfilt(sos, np.asarray(ref, dtype=np.float64)),
    }
    views = (("err", "primary"), ("ref", "primary"))
    view_ratios: dict[str, float] = {}
    view_objectives: dict[str, float] = {}
    for microphone, path in views:
        ratio, objective = causal_v4._estimate_rate_ratio(
            plan=plan,
            submitted=submitted,
            signals={microphone: signals[microphone]},
            method="linear",
            views=((microphone, path),),
        )
        view_ratios[microphone] = ratio
        view_objectives[microphone] = objective
    analysed_frames = int(plan["analysis_stop_sample"]) - int(
        plan["analysis_start_sample"]
    )
    view_disagreement = (max(view_ratios.values()) - min(view_ratios.values())) * analysed_frames
    if view_disagreement > causal_v4.CLOCK_VIEW_DISAGREEMENT_MAX:
        raise ValueError("runtime ERR/REF continuous pilot clock map이 동의하지 않습니다")

    linear_ratio, linear_objective = causal_v4._estimate_rate_ratio(
        plan=plan,
        submitted=submitted,
        signals=signals,
        method="linear",
        views=views,
    )
    cubic_ratio, cubic_objective = causal_v4._estimate_rate_ratio(
        plan=plan,
        submitted=submitted,
        signals=signals,
        method="cubic",
        views=views,
    )
    linear_validation = causal_v4._validate_clock_rows(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=linear_ratio,
        method="linear",
        views=views,
    )
    cubic_validation = causal_v4._validate_clock_rows(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=linear_ratio,
        method="cubic",
        views=views,
    )
    leaveout = float(linear_validation["maximum_leaveout_residual_samples"])
    cubic = max(
        abs(linear_ratio - cubic_ratio) * analysed_frames,
        abs(
            float(cubic_validation["maximum_leaveout_residual_samples"])
            - leaveout
        ),
    )
    combined = leaveout + cubic
    if leaveout > causal_v4.CLOCK_LEAVEOUT_MAX:
        raise ValueError("runtime continuous pilot leaveout residual이 한계를 넘습니다")
    if (
        cubic > causal_v4.CLOCK_CUBIC_MAX
        or combined > causal_v4.CLOCK_COMBINED_MAX
        or combined > causal_v4.CLOCK_HARD_MAX
    ):
        raise ValueError("runtime continuous pilot이 11.314kHz 20dB timing budget을 넘습니다")
    stationarity = _row_stationarity(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=linear_ratio,
    )
    return {
        "rate_ratio_dac_q_per_adc_sample": float(linear_ratio),
        "rate_ppm": float((linear_ratio - 1.0) * 1.0e6),
        "view_rate_ratios": view_ratios,
        "view_objectives": view_objectives,
        "view_end_to_end_disagreement_samples": float(view_disagreement),
        "linear_objective": float(linear_objective),
        "cubic_objective": float(cubic_objective),
        "leaveout_max_samples": leaveout,
        "linear_cubic_max_samples": float(cubic),
        "combined_max_samples": float(combined),
        "hard_20db_11314hz_max_samples": causal_v4.CLOCK_HARD_MAX,
        "minimum_transfer_coherence": min(
            float(linear_validation["minimum_transfer_coherence"]),
            float(cubic_validation["minimum_transfer_coherence"]),
            float(stationarity["minimum_transfer_coherence"]),
        ),
        "submitted_pilot_spectra_sha256": linear_validation[
            "submitted_pilot_spectra_sha256"
        ],
        "segment_stationarity": stationarity,
        "highband_target_or_attenuation_used_for_clock_fit": False,
        "fixed_lti_hypothesis_required": True,
    }


def audit_runtime_physical_clock_witness(
    *,
    plan_path: str | Path,
    session_npz_path: str | Path,
    clock_receipt_path: str | Path,
) -> dict[str, Any]:
    """runtime raw/clock sidecar/pilot plan을 다시 계산해 조건부 판정한다."""

    plan_file = _resolved(plan_path)
    session_file = _resolved(session_npz_path)
    clock_file = _resolved(clock_receipt_path)
    evidence: dict[str, Any] = {
        "schema_version": RUNTIME_PHYSICAL_WITNESS_SCHEMA,
        "status": "BLOCKED",
        "conditional_physical_timing_pass": False,
        "independent_clock_authority_pass": False,
        "canonical_runtime_pass": False,
        "deployment_eligible": False,
        "clock_telemetry_authority_remains": "INCONCLUSIVE",
        "fixed_lti_scope_limitation": _FIXED_LTI_SCOPE,
        "blockers": [],
        "plan_path": str(plan_file),
        "session_npz_path": str(session_file),
        "clock_receipt_path": str(clock_file),
    }
    try:
        plan_raw = _load_json(plan_file)
        plan = _validate_plan(plan_raw)
        if _resolved(plan["session_npz_target"]) != session_file:
            raise ValueError("predeclared session NPZ target과 actual path가 다릅니다")
        if _resolved(plan["clock_receipt_target"]) != clock_file:
            raise ValueError("predeclared clock receipt target과 actual path가 다릅니다")
        plan_file_sha = sha256_file(plan_file)
        session_sha = sha256_file(session_file)
        clock_file_sha = sha256_file(clock_file)
        bundle = _load_json(clock_file)
        telemetry, callbacks, backlog_receipt = _validate_clock_bundle(
            bundle=bundle,
            session_path=session_file,
            session_sha256=session_sha,
        )
        telemetry_sha = payload_sha256(telemetry)
        arrays = _load_session_arrays(session_file, telemetry_sha)

        if not str(telemetry["input_device"]).startswith(
            str(plan["expected_input_device_prefix"])
        ):
            raise ValueError("runtime input device가 predeclared device와 다릅니다")
        if not str(telemetry["output_device"]).startswith(
            str(plan["expected_output_device_prefix"])
        ):
            raise ValueError("runtime output device가 predeclared device와 다릅니다")

        stop = int(plan["analysis_stop_sample"])
        start = int(plan["analysis_start_sample"])
        if min(value.size for value in arrays.values()) < stop:
            raise ValueError("runtime raw가 predeclared 30초 witness window보다 짧습니다")
        gain = arrays["anc_gain"][start:stop]
        full_gain_samples = int(np.count_nonzero(gain >= np.float32(0.999)))
        if full_gain_samples != gain.size:
            raise ValueError("predeclared witness window 전체에서 ANC gain이 full이 아닙니다")
        if full_gain_samples / causal_v4.FS < MINIMUM_WITNESS_SECONDS:
            raise ValueError("full ANC gain runtime witness가 30초 미만입니다")
        if float(np.max(np.abs(arrays["err"][start:stop]))) >= 0.98:
            raise ValueError("runtime ERR raw clipping이 감지됐습니다")
        if float(np.max(np.abs(arrays["ref"][start:stop]))) >= 0.98:
            raise ValueError("runtime REF raw clipping이 감지됐습니다")

        source_pcm = float32_to_pcm_int16(arrays["source"])
        control_pcm = float32_to_pcm_int16(arrays["control"])
        submitted = np.column_stack((source_pcm, control_pcm)).astype(
            np.int16, copy=False
        )
        pilot_receipt = _validate_actual_reserved_pilot(
            submitted=submitted, plan=plan
        )
        callback_receipt = causal_v4._validate_callbacks(callbacks, stop)
        fit = _run_clock_fit(
            plan=plan,
            submitted=submitted,
            err=arrays["err"],
            ref=arrays["ref"],
        )

        synthetic = bool(plan["synthetic_fixture"])
        evidence.update(
            {
                "status": (
                    "FIXTURE_ONLY_PASS" if synthetic else "CONDITIONAL_PASS"
                ),
                "conditional_physical_timing_pass": not synthetic,
                "synthetic_fixture": synthetic,
                "capture_origin": plan["capture_origin"],
                "plan_sha256": plan["plan_sha256"],
                "plan_file_sha256": plan_file_sha,
                "session_npz_sha256": session_sha,
                "clock_receipt_file_sha256": clock_file_sha,
                "clock_telemetry_payload_sha256": telemetry_sha,
                "hardware_fingerprint_sha256": plan[
                    "hardware_fingerprint_sha256"
                ],
                "observed_seconds": (stop - start) / causal_v4.FS,
                "full_anc_gain_seconds": full_gain_samples / causal_v4.FS,
                "source_float32_sha256": _array_sha256(arrays["source"]),
                "control_float32_sha256": _array_sha256(arrays["control"]),
                "source_submitted_pcm_sha256": _array_sha256(source_pcm),
                "control_submitted_pcm_sha256": _array_sha256(control_pcm),
                "stereo_submitted_pcm_sha256": _array_sha256(submitted),
                "submitted_pcm_derivation": (
                    "exact runtime float32_to_pcm_int16: nan_to_num, clip[-1,1], "
                    "rint(value*32767), int16"
                ),
                "raw_err_sha256": _array_sha256(arrays["err"]),
                "raw_ref_sha256": _array_sha256(arrays["ref"]),
                "anc_gain_sha256": _array_sha256(arrays["anc_gain"]),
                "reserved_pilot_receipt": pilot_receipt,
                "callback_witness": callback_receipt,
                "ring_backlog_witness": backlog_receipt,
                "clock_fit": fit,
                "ns_and_cs_share_one_dac_clock": True,
                "adc_dac_drift_is_not_ns_cs_relative_phase_drift": True,
                "highband_target_or_attenuation_used_for_clock_fit": False,
                "pilot_band_is_clock_witness_not_control_or_evaluation_band": True,
                "control_attenuation_assessed": False,
                "octave_125_hz_band_hz": [
                    125.0 / math.sqrt(2.0),
                    125.0 * math.sqrt(2.0),
                ],
                "octave_125_hz_fully_assessed": False,
                "point_control_union_150_11314_assessed": False,
                "independent_electrical_clock_witness_present": False,
                "blockers": (
                    [
                        "synthetic fixture는 physical/canonical runtime PASS로 승격할 수 없습니다"
                    ]
                    if synthetic
                    else []
                ),
            }
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        evidence["blockers"] = [f"{type(exc).__name__}: {exc}"]
        evidence["synthetic_fixture"] = None

    receipt_hash_input = dict(evidence)
    evidence["evidence_sha256"] = _json_sha256(receipt_hash_input)
    return evidence


def write_runtime_physical_witness_receipt_exclusive(
    path: str | Path, evidence: Mapping[str, Any]
) -> tuple[Path, str]:
    """감사 결과를 no-replace로 저장하고 false physical PASS를 막는다."""

    payload = dict(evidence)
    if payload.get("schema_version") != RUNTIME_PHYSICAL_WITNESS_SCHEMA:
        raise ValueError("runtime physical witness evidence schema가 다릅니다")
    status = payload.get("status")
    if status not in {"BLOCKED", "FIXTURE_ONLY_PASS", "CONDITIONAL_PASS"}:
        raise ValueError("runtime physical witness status가 잘못됐습니다")
    if bool(payload.get("synthetic_fixture")) and status == "CONDITIONAL_PASS":
        raise ValueError("synthetic fixture는 physical CONDITIONAL_PASS를 발행할 수 없습니다")
    expected_sha = _require_sha256(
        payload.get("evidence_sha256"), "evidence_sha256"
    )
    unhashed = dict(payload)
    unhashed.pop("evidence_sha256", None)
    if _json_sha256(unhashed) != expected_sha:
        raise ValueError("runtime physical witness evidence SHA가 다릅니다")
    bundle = {
        "schema_version": RUNTIME_PHYSICAL_WITNESS_BUNDLE_SCHEMA,
        "status": status,
        "evidence_sha256": expected_sha,
        "evidence": payload,
    }
    return _write_json_exclusive(path, bundle)


__all__ = [
    "MAX_ROW_GAIN_DEVIATION_DB",
    "MINIMUM_PERIOD_COUNT",
    "MINIMUM_WITNESS_SECONDS",
    "RUNTIME_PHYSICAL_WITNESS_BUNDLE_SCHEMA",
    "RUNTIME_PHYSICAL_WITNESS_PLAN_SCHEMA",
    "RUNTIME_PHYSICAL_WITNESS_SCHEMA",
    "audit_runtime_physical_clock_witness",
    "build_runtime_physical_witness_plan",
    "write_runtime_physical_witness_plan_exclusive",
    "write_runtime_physical_witness_receipt_exclusive",
]
